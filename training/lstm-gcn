import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F

# Data Loading Functions (from your provided code)
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

# LSTM-GCN Model
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TemporalEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # x: [n, t, d]
        return h_n.squeeze(0)  # [n, h]


class GraphConvolutionalLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionalLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        Args:
            x: [num_nodes, in_features] - Node features.
            adj: [num_nodes, num_nodes] - Adjacency matrix.

        Returns:
            output: [num_nodes, out_features] - Updated node features.
        """

        print(f"Adjacency matrix shape: {adj.shape}")
        print(f"Feature tensor shape: {x.shape}")

        # Align dimensions
        if x.size(1) > adj.size(1):
            x = x[:, :adj.size(1), :]  # Trim feature tensor to match adj
        elif x.size(1) < adj.size(1):
            adj = adj[:, :x.size(1), :x.size(1)]  # Trim adjacency matrix

        x = torch.matmul(adj, x)  # Aggregate neighbor features
        x = self.fc(x)  # Apply linear transformation
        return F.relu(x)  # Apply non-linearity


class LSTM_GCN_Model(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, gcn_hidden_dim, output_dim):
        super(LSTM_GCN_Model, self).__init__()
        self.temporal_encoder = TemporalEncoder(input_dim, lstm_hidden_dim)
        self.graph_conv = GraphConvolutionalLayer(lstm_hidden_dim, gcn_hidden_dim)
        self.fc = nn.Linear(gcn_hidden_dim, output_dim)
        # self.num_nodes = num_nodes  # Define number of nodes here
    
    def forward(self, x, edge_index):
        batch_size, num_nodes, num_timesteps, input_dim = x.shape
        
        # LSTM forward pass
        x_flat = x.view(batch_size * num_nodes, num_timesteps, input_dim)
        lstm_out, _ = self.temporal_encoder(x_flat)  # [batch_size * num_nodes, num_timesteps, lstm_hidden_dim]
        
        # Reshape back to [batch_size, num_nodes, lstm_hidden_dim]
        lstm_out = lstm_out.view(batch_size, num_nodes, -1)
        
        # Graph Convolution Layer
        graph_features = self.graph_conv(lstm_out, edge_index)  # [batch_size, num_nodes, gcn_hidden_dim]
        
        # Output Layer
        output = self.fc(graph_features)  # [batch_size, num_nodes, output_dim]
        
        return output


# Training
def train(model, data_loader, relation_tensor, optimizer, criterion, device):
    """
    Trains the LSTM-GCN model.

    Args:
        model: The LSTM-GCN model.
        data_loader: DataLoader providing batches of data.
        relation_tensor: Adjacency matrix representing relationships between nodes.
        optimizer: Optimizer for training (e.g., Adam).
        criterion: Loss function (e.g., MSELoss).
        device: Device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        avg_loss: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in data_loader:
        eod_data, targets = batch  # Unpack batch
        eod_data = eod_data.to(device)  # [batch_size, num_nodes, num_timesteps, input_dim]
        relation_tensor = relation_tensor.to(device)  # [num_nodes, num_nodes]
        targets = targets.to(device)  # [batch_size, num_nodes, output_dim]

        optimizer.zero_grad()

        # Forward pass
        predictions = model(eod_data, relation_tensor)  # [batch_size, num_nodes, output_dim]

        # Ensure predictions and targets have the same shape
        if predictions.shape != targets.shape:
            # Expand targets to match predictions if needed
            targets = targets.unsqueeze(1).expand_as(predictions)

        # Compute loss
        loss = criterion(predictions, targets)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

# Main Function
def main():
    # Hyperparameters
    input_dim = 5  # Number of features per stock
    lstm_hidden_dim = 64
    gcn_hidden_dim = 32
    output_dim = 1  # Predicting 1 value (e.g., price)
    learning_rate = 0.001
    epochs = 50
    batch_size = 32

    # Paths
    data_path = r"C:\Users\palak\Desktop\.vscode\quant whitepaper\Temporal_Relational_Stock_Ranking\data\2013-01-01"
    relation_file = r"C:\Users\palak\Desktop\.vscode\quant whitepaper\Temporal_Relational_Stock_Ranking\data\relation\sector_industry\NASDAQ_industry_relation.npy"
    market_name = "NASDAQ"
    tickers = np.genfromtxt(r"C:\Users\palak\Desktop\.vscode\quant whitepaper\Temporal_Relational_Stock_Ranking\data\NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv",
                                     dtype=str, delimiter='\t', skip_header=False)  # Example tickers

    # Load data
    eod_data, _, ground_truth, _ = load_EOD_data(data_path, market_name, tickers)
    relation_matrix = load_graph_relation_data(relation_file)


    # Prepare PyTorch tensors
    eod_tensor = torch.tensor(eod_data, dtype=torch.float32)
    relation_tensor = torch.tensor(relation_matrix, dtype=torch.float32)
    target_tensor = torch.tensor(ground_truth, dtype=torch.float32)

    # # Reshape relation_tensor to match timesteps
    # relation_tensor = relation_tensor.unsqueeze(0).repeat(eod_tensor.shape[1], 1, 1)  # Shape: [num_timesteps, num_stocks, num_stocks]

    # # Adjust target_tensor to match eod_tensor
    # target_tensor = target_tensor.permute(1, 0)  # Shape: [num_timesteps, num_stocks]

    # Dataset and DataLoader
    dataset = torch.utils.data.TensorDataset(eod_tensor, target_tensor)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Optimizer, and Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_GCN_Model(input_dim, lstm_hidden_dim, gcn_hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training Loop
    for epoch in range(epochs):
        loss = train(model, data_loader, relation_tensor, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
