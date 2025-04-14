import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np

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
        'tavuk':10,
        'chicken':10
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
            label = LABELS[str(file_name.split('_')[0]).lower()]  # Extract label from file name
            self.data.append(tuple([torch.tensor(list(data)),label]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# Define transformations for the training, validation, and testing sets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = CustomDataset(root_dir='../pdata/waveforms/training', transform=transform)
val_dataset = CustomDataset(root_dir='../pdata/waveforms/validation', transform=transform)
test_dataset = CustomDataset(root_dir='../pdata/waveforms/testing', transform=transform)

# Create data loaders
'''
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
'''

# Define the 1D CNN model
class Simple1DCNN(nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 64, 512)  # Adjust based on input length
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64)  # Adjust based on input length
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
    for data, labels in train_dataset.data:
        data = data.unsqueeze(1)  # Add channel dimension
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
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

