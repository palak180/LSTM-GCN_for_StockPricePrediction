import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# Data Loading Functions 
def load_EOD_data(data_path, market_name, tickers, steps=1):
    eod_data = []
    masks = []
    ground_truth = []
    base_price = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ':
            single_EOD = single_EOD[:-1, :]
        if index == 0:
            eod_data = np.zeros([len(tickers), single_EOD.shape[0], single_EOD.shape[1] - 1], dtype=np.float32)
            masks = np.ones([len(tickers), single_EOD.shape[0]], dtype=np.float32)
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]], dtype=np.float32)
            base_price = np.zeros([len(tickers), single_EOD.shape[0]], dtype=np.float32)
        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                masks[index][row] = 0.0
            elif row > steps - 1 and abs(single_EOD[row - steps][-1] + 1234) > 1e-8:
                ground_truth[index][row] = (single_EOD[row][-1] - single_EOD[row - steps][-1]) / single_EOD[row - steps][-1]
            for col in range(single_EOD.shape[1]):
                if abs(single_EOD[row][col] + 1234) < 1e-8:
                    single_EOD[row][col] = 1.1
        eod_data[index, :, :] = single_EOD[:, 1:]
        base_price[index, :] = single_EOD[:, -1]
    return eod_data, masks, ground_truth, base_price


def load_graph_relation_data(relation_file, lap=False):
    relation_encoding = np.load(relation_file)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int), np.sum(relation_encoding, axis=2))
    ajacent = np.where(mask_flags, np.zeros(rel_shape, dtype=float), np.ones(rel_shape, dtype=float))
    degree = np.sum(ajacent, axis=0)
    for i in range(len(degree)):
        degree[i] = 1.0 / degree[i]
    np.sqrt(degree, degree)
    deg_neg_half_power = np.diag(degree)
    if lap:
        return np.identity(ajacent.shape[0], dtype=float) - np.dot(np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)
    else:
        return np.dot(np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)



def train_test_split(eod_tensor, target_tensor, split_ratio=0.8):
    """Splits data into training and testing sets."""
    num_samples = eod_tensor.shape[0]
    split_idx = int(num_samples * split_ratio)

    train_eod = eod_tensor[:split_idx]
    test_eod = eod_tensor[split_idx:]

    train_targets = target_tensor[:split_idx]
    test_targets = target_tensor[split_idx:]

    return (train_eod, train_targets), (test_eod, test_targets)



class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, features, relation_tensor):
        """
        features: [num_nodes, in_features]
        relation_tensor: [num_nodes, num_nodes]
        """
        # Normalize adjacency matrix
        degree = torch.sum(relation_tensor, dim=1).clamp(min=1.0)  # Prevent division by zero
        norm_relation_tensor = relation_tensor / degree.unsqueeze(1)

        print(f"Features shape: {features.shape}")  # Debug print
        print(f"Relation tensor shape: {relation_tensor.shape}")  # Debug print
        print(f"Normalized relation tensor shape: {norm_relation_tensor.shape}")  # Debug print

        # Graph convolution
        aggregated_features = torch.matmul(norm_relation_tensor, features)  # [num_nodes, in_features]
        updated_features = self.linear(aggregated_features)  # [num_nodes, out_features]
        return F.relu(updated_features)


class GCN_LSTM(nn.Module):
    def __init__(self, num_nodes, num_features, gcn_hidden_dim, lstm_hidden_dim, output_dim):
        super(GCN_LSTM, self).__init__()
        self.gcn_input = GCNLayer(num_features, gcn_hidden_dim)
        self.gcn_hidden = GCNLayer(lstm_hidden_dim, gcn_hidden_dim)
        self.gcn_cell = GCNLayer(lstm_hidden_dim, gcn_hidden_dim)
        self.lstm = nn.LSTM(gcn_hidden_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, eod_tensor, relation_tensor):
        """
        eod_tensor: [num_timesteps, num_nodes, num_features]
        relation_tensor: [num_nodes, num_nodes]
        """
        num_timesteps, num_nodes, num_features = eod_tensor.shape

        # Prepare LSTM initial states
        h_t = torch.zeros(1, 1, self.lstm.hidden_size, device=eod_tensor.device)  # [num_layers, batch_size, lstm_hidden_dim]
        c_t = torch.zeros(1, 1, self.lstm.hidden_size, device=eod_tensor.device)  # [num_layers, batch_size, lstm_hidden_dim]

        lstm_outputs = []

        for t in range(num_timesteps):
            # GCN on input features
            gcn_input_features = self.gcn_input(eod_tensor[t], relation_tensor)  # [num_nodes, gcn_hidden_dim]

            # GCN on hidden state
            gcn_hidden_state = self.gcn_hidden(h_t.squeeze(0), relation_tensor)  # [num_nodes, gcn_hidden_dim]

            # GCN on cell state
            gcn_cell_state = self.gcn_cell(c_t.squeeze(0), relation_tensor)  # [num_nodes, gcn_hidden_dim]

            # Combine GCN outputs for input to LSTM
            combined_features = gcn_input_features  # [num_nodes, gcn_hidden_dim]

            # Aggregate GCN-processed hidden states to match LSTM input
            lstm_h_t = gcn_hidden_state.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, gcn_hidden_dim]
            lstm_c_t = gcn_cell_state.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, gcn_hidden_dim]

            # Reshape for LSTM: [batch_size=1, seq_len=num_nodes, input_dim=gcn_hidden_dim]
            combined_features = combined_features.unsqueeze(0)  # Add batch dimension for LSTM

            print(f"combined_features shape: {combined_features.shape}")
            print(f"lstm_h_t shape: {lstm_h_t.shape}, lstm_c_t shape: {lstm_c_t.shape}")

            # LSTM step
            lstm_out, (h_t, c_t) = self.lstm(combined_features, (lstm_h_t, lstm_c_t))

            lstm_outputs.append(lstm_out.squeeze(0))  # Append output for this time step

        # Stack LSTM outputs: [num_timesteps, num_nodes, lstm_hidden_dim]
        lstm_outputs = torch.stack(lstm_outputs, dim=0)

        # Predict next day's closing prices
        predictions = self.fc(lstm_outputs[-1])  # Use the last time step's output
        return predictions

def train(model, data_loader, relation_tensor, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in data_loader:
        eod_data, targets = batch  # Unpack batch
        eod_data = eod_data.to(device)  # [batch_size, num_nodes, num_timesteps, input_dim]
        relation_tensor = relation_tensor.to(device)  # [num_nodes, num_nodes]
        targets = targets.to(device)  # [batch_size, num_nodes, output_dim]

        optimizer.zero_grad()

        # Squeeze the timestep dimension
        eod_data = eod_data.squeeze(2)  # [batch_size, num_nodes, input_dim]
        print(f"Relation tensor shape: {relation_tensor.shape}")
        print(f"eod_data shape after squeeze: {eod_data.shape}")

        # print("EOD_data:", eod_data.shape)
        # Forward pass
        predictions = model(eod_data, relation_tensor)  # [batch_size, num_nodes, output_dim]

        # Ensure predictions and targets have the same shape
        targets = targets.view_as(predictions)  # Adjust shape to match predictions

        # Compute loss
        loss = criterion(predictions, targets)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def evaluate(model, data_loader, relation_tensor, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            eod_data, targets = batch  # Unpack batch
            eod_data = eod_data.to(device)
            relation_tensor = relation_tensor.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(eod_data, relation_tensor)

            # Ensure predictions and targets have the same shape
            targets = targets.view_as(predictions)

            # Compute loss
            loss = criterion(predictions, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def plot_actual_vs_predicted(model, data_loader, relation_tensor, device, save_path="actual_vs_predicted.png"):
    model.eval()
    actuals = []
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            eod_data, targets = batch
            eod_data = eod_data.to(device)
            relation_tensor = relation_tensor.to(device)
            targets = targets.to(device)

            preds = model(eod_data, relation_tensor)
            actuals.append(targets.cpu().numpy())
            predictions.append(preds.cpu().numpy())

    actuals = np.concatenate(actuals, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(actuals.flatten(), label="Actual Prices", marker='o')
    plt.plot(predictions.flatten(), label="Predicted Prices", marker='x')
    plt.xlabel("Time Steps")
    plt.ylabel("Closing Prices")
    plt.title("Actual vs Predicted Closing Prices")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Hyperparameters
    input_dim = 5
    lstm_hidden_dim = 64
    gcn_hidden_dim = 32
    output_dim = 1
    learning_rate = 0.001
    epochs = 10
    batch_size = 32

    # Paths
    data_path = r"./data/2013-01-01"
    relation_file = r"./data/relation/sector_industry/NYSE_industry_relation.npy"
    market_name = "NYSE"
    tickers = np.genfromtxt(r"./data/NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv",
                             dtype=str, delimiter='\t', skip_header=False)
    print(f"Number of tickers: {len(tickers)}")

    # Load data
    eod_data, _, ground_truth, _ = load_EOD_data(data_path, market_name, tickers)
    relation_matrix = np.load(relation_file)  # Assuming this is pre-normalized

    print(f"Original EOD shape: {eod_data.shape}")

    # Prepare tensors
    eod_tensor = torch.tensor(eod_data, dtype=torch.float32).permute(1, 0, 2).unsqueeze(2)
    relation_tensor = torch.tensor(relation_matrix, dtype=torch.float32)
    target_tensor = torch.tensor(ground_truth, dtype=torch.float32).permute(1, 0).unsqueeze(2)

    # Split data
    (train_eod, train_targets), (test_eod, test_targets) = train_test_split(eod_tensor, target_tensor)

    print('train_eod shape:', train_eod.shape)
    train_dataset = TensorDataset(train_eod, train_targets)
    test_dataset = TensorDataset(test_eod, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    relation_tensor = relation_tensor.sum(dim=-1)

    # Model, Optimizer, Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN_LSTM(len(tickers), input_dim, gcn_hidden_dim, lstm_hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Train
        train_loss = train(model, train_loader, relation_tensor, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Test
        test_loss = evaluate(model, test_loader, relation_tensor, criterion, device)
        test_losses.append(test_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # Plot train and test loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Test Loss vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_test_loss_vs_epochs.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot Actual vs Predicted Targets
    plot_actual_vs_predicted(model, test_loader, relation_tensor, device, save_path="actual_vs_predicted.png")


if __name__ == "__main__":
    main()


