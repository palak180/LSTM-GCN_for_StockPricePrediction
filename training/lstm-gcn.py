import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os

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
            eod_data = np.zeros([len(tickers), single_EOD.shape[0], single_EOD.shape[1] - 2], dtype=np.float32)
            masks = np.ones([len(tickers), single_EOD.shape[0]], dtype=np.float32)
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]], dtype=np.float32)
            base_price = np.zeros([len(tickers), single_EOD.shape[0]], dtype=np.float32)
        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                masks[index][row] = 0.0
            elif row > steps - 1 and abs(single_EOD[row - steps][-1] + 1234) > 1e-8:
                ground_truth[index][row] = single_EOD[row][-1]
            for col in range(single_EOD.shape[1]):
                if abs(single_EOD[row][col] + 1234) < 1e-8:
                    single_EOD[row][col] = 1.1
        eod_data[index, :, :] = single_EOD[:, 1:-1]
        base_price[index, :] = single_EOD[:, -1]
    return eod_data, masks, ground_truth, base_price


def load_graph_relation_data(relation_file, lap=False):
    relation_encoding = np.load(relation_file)
    adj_matrix = np.sum(relation_encoding, axis=2)
    mask_flags = np.equal(adj_matrix, 0)
    removed_nodes = np.where(np.sum(mask_flags, axis=1) == adj_matrix.shape[1])[0]
    adj_matrix = np.where(mask_flags, 0, adj_matrix)

    # Randomly select 100 nodes
    all_nodes = np.arange(adj_matrix.shape[0])
    random_nodes = np.random.choice(all_nodes, size=100, replace=False)

    # Subset the adjacency matrix to only include the selected nodes
    adj_matrix = adj_matrix[np.ix_(random_nodes, random_nodes)]

    # Degree normalization
    degree = np.sum(adj_matrix, axis=0)
    degree[degree != 0] = 1.0 / degree[degree != 0]  # Avoid division by zero
    degree_sqrt = np.sqrt(degree)

    # Create degree matrix
    deg_neg_half_power = np.diag(degree_sqrt)

    if lap:
        # Return Laplacian normalized adjacency matrix
        normalized_matrix = (
            np.identity(adj_matrix.shape[0], dtype=float) - 
            np.dot(np.dot(deg_neg_half_power, adj_matrix), deg_neg_half_power)
        )
    else:
        # Return normalized adjacency matrix
        normalized_matrix = np.dot(np.dot(deg_neg_half_power, adj_matrix), deg_neg_half_power)

    return normalized_matrix, removed_nodes, random_nodes

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X, A):
        """
        X: Input features (num_nodes x in_features)
        A: Normalized Adjacency matrix (num_nodes x num_nodes)
        """
        return A @ X @ self.weight

class GCN_LSTM(nn.Module):
    def __init__(self, num_nodes, input_features, hidden_features, output_features):
        super(GCN_LSTM, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_features = hidden_features

        # GCN Layers
        self.gcn_h = GCNLayer(hidden_features, hidden_features)
        self.gcn_c = GCNLayer(hidden_features, hidden_features)
        self.gcn_x = GCNLayer(input_features, hidden_features)

        # LSTM Cell
        self.lstm_cell = nn.LSTMCell(hidden_features, hidden_features)

        # Output layer
        self.fc_out = nn.Linear(hidden_features, output_features)

    def forward(self, X, A):
        """
        X: Input features (rolling_window_size x num_nodes x input_features)
        A: Adjacency matrix (num_nodes x num_nodes)
        """
        rolling_window_size, num_nodes, input_features = X.size()
        h = torch.zeros(num_nodes, self.hidden_features).to(X.device)
        c = torch.zeros(num_nodes, self.hidden_features).to(X.device)

        for t in range(rolling_window_size):
            # Apply GCN on h(t-1), c(t-1), and X(t)
            h_gcn = self.gcn_h(h, A)
            c_gcn = self.gcn_c(c, A)
            x_gcn = self.gcn_x(X[t], A)

            # Update h and c using LSTMCell
            h, c = self.lstm_cell(x_gcn + h_gcn + c_gcn, (h, c))

        # Output prediction for the next timestep
        output = self.fc_out(h)
        return output

# Training Loop
def train_model(model, data, price_data, adj_matrix, window_size, criterion, optimizer, num_epochs=10):
    """
    model: GCN-LSTM model
    data: EOD data (num_nodes x num_timesteps x num_features)
    price_data: Ground truth price data (num_nodes x num_timesteps)
    adj_matrix: Adjacency matrix (num_nodes x num_nodes)
    window_size: Number of timesteps in the rolling window
    criterion: Loss function
    optimizer: Optimizer
    num_epochs: Number of training epochs
    """
    print("eod_tensor:", data.shape, "target_tensor:", price_data.shape, "relation_tensor:", adj_matrix.shape)

    num_nodes, num_timesteps, num_features = data.size()
    losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        for t in range(num_timesteps - window_size - 1):
            # Slice rolling window
            X = data[:, t:t+window_size, :]  # Input: 100 timesteps
            y = price_data[:, t+window_size]  # Target: 101st timestep (price)

            # Reshape for the model
            X = X.permute(1, 0, 2)  # (rolling_window_size, num_nodes, num_features)
            y = y.view(num_nodes, -1)  # (num_nodes, 1)

            # Forward pass
            output = model(X, adj_matrix)  # (num_nodes, output_features)

            # Compute loss
            loss = criterion(output.squeeze(), y.squeeze())
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / (num_timesteps - window_size - 1)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.8f}")

    return losses

# Example Usage
# input_features = 10  # Example input features per node
hidden_features = 32
output_features = 1  # Predicting one value per node (e.g., stock price)

# Paths
data_path = r"./data/2013-01-01"
relation_file = r"./data/relation/sector_industry/NYSE_industry_relation.npy"
market_name = "NYSE"
tickers = np.genfromtxt(r"./data/NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv",
                            dtype=str, delimiter='\t', skip_header=False)

num_nodes = len(tickers)

# Load data
relation_matrix, removed_nodes, selected_nodes = load_graph_relation_data(relation_file)  # Assuming this is pre-normalized
print(len(removed_nodes))

tickers = tickers[selected_nodes]
eod_data, _, ground_truth, _ = load_EOD_data(data_path, market_name, tickers)

input_features = eod_data.shape[2]
rolling_window_size = 200

# Prepare tensors
eod_tensor = torch.tensor(eod_data, dtype=torch.float32)
relation_tensor = torch.tensor(relation_matrix, dtype=torch.float32)
target_tensor = torch.tensor(ground_truth, dtype=torch.float32)

eod_tensor = eod_tensor[:, -400:, :]  # Keep all nodes and features
target_tensor = target_tensor[:, -400:]  # Keep all nodes, only price data

model = GCN_LSTM(num_nodes, input_features, hidden_features, output_features)
optimizer = optim.Adam(model.parameters(), lr = 0.0015)
criterion = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
eod_tensor = eod_tensor.to(device)
target_tensor = target_tensor.to(device)
relation_tensor = relation_tensor.to(device)

train_model(model, eod_tensor, target_tensor, relation_tensor, rolling_window_size, criterion, optimizer)
