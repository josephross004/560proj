import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np

print("starting: dependencies imported.")

LABELS = {
        'aslan':1,
        'lion':1,
        'esek':2,
        'donkey':2,
        'inek':3,
        'cow':3,
        'kedi':4,
        'cat':4,
        'kopek':5,
        'dog':5,
        'koyun':6,
        'sheep':6,
        'kurbaga':7,
        'frog':7,
        'kus':8,
        'bird':8,
        'maymun':9,
        'monkey':9,
        'tavuk':0,
        'chicken':0
        }

# Custom dataset class to handle varying lengths of 1D data with labels in file names
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.load_data()

    def load_data(self):
        for file_name in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file_name)
            data = np.load(file_path)  # Assuming data is stored in .npy format
            label_str = file_name.split('_')[0].lower()
            if label_str in LABELS:
                label = LABELS[label_str]
                self.data.append(tuple([data, label]))
            else:
                print(f"Warning: Label '{label_str}' from file '{file_name}' not found in LABELS.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx]
        if self.transform:
            data = self.transform(data)
        return data.float(), label

class To1DTensor(object):
    """Convert ndarrays in sample to Tensors and add channel dim."""
    def __call__(self, sample):
        tensor_sample = torch.from_numpy(sample).float().unsqueeze(0)
        print(f"Shape inside To1DTensor: {tensor_sample.shape}")
        return tensor_sample

class Normalize1D(object):
    def __call__(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-8)


# Define transformations for the training, validation, and testing sets
transform = transforms.Compose([
    To1DTensor(),
    Normalize1D()
])

transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
print("load: training...")
train_dataset = CustomDataset(root_dir='./pdata/waveforms/training', transform=transform)
print("done.")
print("load: validation...")
val_dataset = CustomDataset(root_dir='./pdata/waveforms/validation', transform=transform)
print("done.")
print("load: testing...")
test_dataset = CustomDataset(root_dir='./pdata/waveforms/testing', transform=transform)
print("done.")

def collate_fn_pad(batch):
    data = [item[0].squeeze(0) for item in batch]  # Squeeze the channel dimension
    label = [item[1] for item in batch]

    # Pad sequences to the maximum length in the batch
    data_padded = pad_sequence(data, batch_first=True, padding_value=0.0)
    label = torch.tensor(label)
    return data_padded, label

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_pad)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_pad)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_pad)

import torch.nn as nn

class Simple1DCNN(nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1) # Output size of 1
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, max_length)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # Global Average Pooling
        x = self.global_avg_pool(x) # Output: (batch_size, 64, 1)
        x = x.squeeze(-1)          # Remove the last dimension: (batch_size, 64)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = Simple1DCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data, label in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, labels in val_dataset.data:
            data = data.unsqueeze(1)  # Add channel dimension
            outputs = model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f'Validation Loss: {val_loss/len(val_loader)}')

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, labels in test_dataset.data:
        data = data.unsqueeze(1)  # Add channel dimension
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total}%')

