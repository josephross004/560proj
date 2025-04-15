import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
from sklearn.metrics import accuracy_score

# Define a custom Dataset class with padding
class PaddedMatrixDataset(Dataset):
    def __init__(self, data_dir, max_cols=None):
        self.data_dir = data_dir
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.labels = [re.match(r'([^_\s]+)', os.path.basename(fp)).group(1) for fp in self.file_paths]
        self.unique_labels = sorted(list(set(self.labels)))
        self.label_to_index = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        self.max_cols = max_cols
        if self.max_cols is None:
            self.max_cols = self._find_max_cols()

    def _find_max_cols(self):
        max_c = 0
        for fp in self.file_paths:
            matrix = np.load(fp)
            max_c = max(max_c, matrix.shape[1])
        return max_c

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        matrix = np.load(file_path)
        label_str = self.labels[idx]
        label_index = self.label_to_index[label_str]

        # Pad the matrix
        num_rows, num_cols = matrix.shape
        if num_cols < self.max_cols:
            padding_width = self.max_cols - num_cols
            padding = np.zeros((num_rows, padding_width), dtype=matrix.dtype)
            padded_matrix = np.hstack((matrix, padding))
        else:
            padded_matrix = matrix

        # Convert numpy array to PyTorch tensor and add a channel dimension
        matrix_tensor = torch.tensor(padded_matrix, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label_index, dtype=torch.long)
        return matrix_tensor, label_tensor

# Define the CNN model (adjusting the first FC layer)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, max_cols):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # Calculate the flattened size after convolution and pooling with fixed max_cols
        # Input: 1x9xmax_cols
        # Conv1: 16x9xmax_cols
        # Pool1: 16x4x(max_cols//2)
        # Conv2: 32x4x(max_cols//2)
        # Pool2: 32x2x(max_cols//4)
        self.fc1 = nn.Linear(32 * 2 * (max_cols // 4), 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Define your data directories
    train_dir = './pdata/spectra/training/'
    val_dir = './pdata/spectra/testing/'
    test_dir = './pdata/spectra/validation/'


    # Determine the maximum number of columns in the training set
    train_dataset_temp = PaddedMatrixDataset(train_dir)
    max_cols = train_dataset_temp.max_cols
    print(f"Maximum number of columns in training data: {max_cols}")

    # Create datasets with padding to the maximum number of columns found in training
    train_dataset = PaddedMatrixDataset(train_dir, max_cols=max_cols)
    val_dataset = PaddedMatrixDataset(val_dir, max_cols=max_cols)
    test_dataset = PaddedMatrixDataset(test_dir, max_cols=max_cols)

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Get the number of unique classes
    num_classes = len(train_dataset.unique_labels)
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {train_dataset.label_to_index}")

    # Instantiate the model, passing max_cols
    model = SimpleCNN(num_classes, max_cols)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Check if CUDA is available and use GPU if it is
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Testing loop
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_epoch_loss = test_loss / len(test_loader)
    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Testing Loss: {test_epoch_loss:.4f}, Testing Accuracy: {test_accuracy:.4f}")

    # You can save the trained model here if needed
    # torch.save(model.state_dict(), 'path/to/save/model.pth')