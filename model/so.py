import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import re

# --- Configuration ---
DATA_DIR = './pdata/spectrograms/'  # Replace with the actual path to your images
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
TRAIN_DIR = os.path.join(DATA_DIR, 'training')
TEST_DIR = os.path.join(DATA_DIR, 'testing')
VAL_DIR = os.path.join(DATA_DIR, 'validation')

# --- Helper Function to Extract Label from Filename ---
def extract_label(filename):
    match = re.match(r'([^_\s]+)', filename)
    if match:
        return match.group(1)
    return None

# --- Custom Dataset Class for Black and White Images ---
class BWImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        self.transform = transform
        self.labels = [extract_label(f) for f in self.img_files]
        self.unique_labels = sorted(list(set(self.labels)))
        self.label_to_index = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('L')  # Open as grayscale ('L')
        label = self.labels[idx]
        label_index = self.label_to_index[label]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_index, dtype=torch.long)

# --- Data Transformations for Black and White Images ---
# Adjust Normalize for single-channel images
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # Single mean and std for grayscale
])

val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # Single mean and std for grayscale
])

# --- Create Datasets and DataLoaders ---
train_dataset = BWImageDataset(TRAIN_DIR, transform=train_transforms)
val_dataset = BWImageDataset(VAL_DIR, transform=val_test_transforms)
test_dataset = BWImageDataset(TEST_DIR, transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Define the CNN Model for Black and White Images ---
class BWCNN(nn.Module):
    def __init__(self, num_classes):
        super(BWCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) # Input channels = 1
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Adjust input size based on image size after pooling
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten the tensor
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Initialize Model, Loss Function, and Optimizer ---
num_classes = len(train_dataset.unique_labels)
model = BWCNN(num_classes) # Use the BWCNN model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(NUM_EPOCHS):
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
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {epoch_loss:.4f}")

    # --- Validation Loop ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Accuracy: {val_accuracy:.2f}%")

print("Finished Training")

# --- Testing Loop ---
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# --- Optional: Save the Trained Model ---
# torch.save(model.state_dict(), 'bw_image_classifier.pth')
# print("Trained model saved as bw_image_classifier.pth")