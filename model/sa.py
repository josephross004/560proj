import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import List, Tuple

class ContourDataset(Dataset):
    """Dataset for the percentile contour matrices."""
    def __init__(self, data_dir: str, max_len: int):
        self.data_dir = data_dir
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.max_len = max_len
        self.labels = self._extract_labels()
        self.data = self._load_and_pad_data()
        self.unique_labels = sorted(list(set(self.labels)))
        self.label_to_index = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.indexed_labels = [self.label_to_index[label] for label in self.labels]

    def _extract_labels(self) -> List[str]:
        """Extracts labels from the file names."""
        return [f.split('_')[0] for f in self.file_names]

    def _load_and_pad_data(self) -> List[np.ndarray]:
        """Loads the numpy matrices and pads them to the maximum length."""
        padded_data = []
        for file_name in self.file_names:
            file_path = os.path.join(self.data_dir, file_name)
            matrix = np.load(file_path)
            current_len = matrix.shape[1]
            if current_len < self.max_len:
                padding_width = self.max_len - current_len
                padding = np.zeros((9, padding_width), dtype=matrix.dtype)
                padded_matrix = np.concatenate((matrix, padding), axis=1)
            elif current_len > self.max_len:
                padded_matrix = matrix[:, :self.max_len]
            else:
                padded_matrix = matrix
            padded_data.append(padded_matrix)
        return padded_data

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        matrix = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.indexed_labels[idx], dtype=torch.long)
        return matrix, label

class SimpleCNN(nn.Module):
    def __init__(self, input_channels: int, seq_len: int, num_classes: int):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Calculate the size of the flattened layer
        self._to_linear = None
        self._determine_linear_size(torch.randn(1, input_channels, seq_len))
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def _determine_linear_size(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        self._to_linear = x.view(x.size(0), -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int, device: str):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

def evaluate_model(model: nn.Module, data_loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':
    # Define your data directories
    train_data_dir = './pdata/spectra/training'
    val_data_dir = './pdata/spectra/validation'
    test_data_dir = './pdata/spectra/testing'

    # Ensure the directories exist (replace with your actual paths)
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(val_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    # Determine the maximum sequence length across all datasets
    def get_max_len(data_dirs):
        max_len = 0
        for data_dir in data_dirs:
            for file_name in os.listdir(data_dir):
                if file_name.endswith('.npy'):
                    matrix = np.load(os.path.join(data_dir, file_name))
                    max_len = max(max_len, matrix.shape[1])
        return max_len

    max_sequence_length = get_max_len([train_data_dir, val_data_dir, test_data_dir])
    print(f"Maximum sequence length: {max_sequence_length}")

    # Create Datasets and DataLoaders
    train_dataset = ContourDataset(train_data_dir, max_sequence_length)
    val_dataset = ContourDataset(val_data_dir, max_sequence_length)
    test_dataset = ContourDataset(test_data_dir, max_sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define the model
    input_channels = 9  # Number of rows in the matrix
    num_classes = len(train_dataset.unique_labels)
    model = SimpleCNN(input_channels, max_sequence_length, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the model
    num_epochs = 10
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # Evaluate on the validation set
    val_accuracy = evaluate_model(model, val_loader, device)
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

    # Evaluate on the test set
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.2f}%')