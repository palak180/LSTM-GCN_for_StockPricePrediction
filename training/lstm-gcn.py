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

def evaluate(model, data_loader, relation_tensor, criterion, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            eod_data, targets = batch
            eod_data = eod_data.to(device)
            relation_tensor = relation_tensor.to(device)
            targets = targets.to(device)

            predictions = model(eod_data, relation_tensor)

            if predictions.shape != targets.shape:
                targets = targets.view_as(predictions)

            loss = criterion(predictions, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


# LSTM-GCN Model
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TemporalEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        print(f"TemporalEncoder Input shape: {x.shape}")
        lstm_out, (h_n, c_n) = self.lstm(x)
        print(f"LSTM Output shape: {lstm_out.shape}")
        print(f"LSTM Hidden State shape: {h_n.shape}")
        print(f"LSTM Cell State shape: {c_n.shape}")
        return h_n[-1]


class GraphConvolutionalLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionalLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        x: Node features with shape [batch_size, num_nodes, input_dim]
        adj: Adjacency matrix with shape [num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = x.shape

        # Expand adjacency matrix to match batch size
        adj = adj.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, num_nodes, num_nodes]

        # Perform batched matrix multiplication
        x = torch.bmm(adj, x)  # Shape: [batch_size, num_nodes, input_dim]

        # Apply the linear transformation
        x = self.fc(x)  # Shape: [batch_size, num_nodes, out_features]

        return x


class LSTM_GCN_Model(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, gcn_hidden_dim, output_dim):
        super(LSTM_GCN_Model, self).__init__()
        self.temporal_encoder = TemporalEncoder(input_dim, lstm_hidden_dim)
        self.graph_conv = GraphConvolutionalLayer(lstm_hidden_dim, gcn_hidden_dim)
        self.fc = nn.Linear(gcn_hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        batch_size, num_nodes, num_timesteps, input_dim = x.shape
        print(f"Forward Input shape: {x.shape}")
        
        # LSTM forward pass
        x_flat = x.view(batch_size * num_nodes, num_timesteps, input_dim)
        lstm_out = self.temporal_encoder(x_flat)  # [batch_size * num_nodes, lstm_hidden_dim]
        print(f"LSTM Output shape received in model: {lstm_out.shape}")
        
        # Reshape back to [batch_size, num_nodes, lstm_hidden_dim]
        lstm_out = lstm_out.view(batch_size, num_nodes, -1)
        
        # Graph Convolution Layer
        graph_features = self.graph_conv(lstm_out, edge_index)  # [batch_size, num_nodes, gcn_hidden_dim]
        print(f"Graph Convolution Output shape: {graph_features.shape}")
        
        # Output Layer
        output = self.fc(graph_features)  # [batch_size, num_nodes, output_dim]
        print(f"Output shape: {output.shape}")
        
        return output


# Training
def train(model, data_loader, relation_tensor, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in data_loader:
        eod_data, targets = batch  # Unpack batch
        eod_data = eod_data.to(device)  # [batch_size, num_nodes, num_timesteps, input_dim]
        relation_tensor = relation_tensor.to(device)  # [num_nodes, num_nodes]
        targets = targets.to(device)  # [batch_size, num_nodes, output_dim]

        print(f"Training batch - eod_data shape: {eod_data.shape}, targets shape: {targets.shape}")
        
        optimizer.zero_grad()

        # Forward pass
        predictions = model(eod_data, relation_tensor)  # [batch_size, num_nodes, output_dim]

        # Ensure predictions and targets have the same shape
        if predictions.shape != targets.shape:
            targets = targets.view_as(predictions)  # Adjust shape to match predictions

        print(f"Predictions shape: {predictions.shape}")

        # Compute loss
        loss = criterion(predictions, targets)
        print(f"Loss: {loss.item()}")

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def main():
    # Hyperparameters
    input_dim = 5
    lstm_hidden_dim = 64
    gcn_hidden_dim = 32
    output_dim = 1
    learning_rate = 0.001
    epochs = 20
    batch_size = 32

    # Paths
    data_path = r"./data/2013-01-01"
    relation_file = r"./data/relation/sector_industry/NASDAQ_industry_relation.npy"
    market_name = "NASDAQ"
    tickers = np.genfromtxt(r"./data/NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv",
                             dtype=str, delimiter='\t', skip_header=False)

    # Load data
    eod_data, _, ground_truth, _ = load_EOD_data(data_path, market_name, tickers)
    relation_matrix = load_graph_relation_data(relation_file)

    # Prepare tensors
    eod_tensor = torch.tensor(eod_data, dtype=torch.float32).permute(1, 0, 2).unsqueeze(2)
    relation_tensor = torch.tensor(relation_matrix, dtype=torch.float32)  # [num_nodes, num_nodes]
    target_tensor = torch.tensor(ground_truth, dtype=torch.float32).permute(1, 0).unsqueeze(2)

    (train_eod, train_targets), (test_eod, test_targets) = train_test_split(eod_tensor, target_tensor)

    train_dataset = TensorDataset(train_eod, train_targets)
    test_dataset = TensorDataset(test_eod, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Optimizer, Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_GCN_Model(input_dim, lstm_hidden_dim, gcn_hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Train
        model.train()
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
    # Save the plot as an image file
    plt.savefig("train_test_loss_vs_epochs.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
