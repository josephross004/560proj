import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import traceback
from torchvision.datasets import DatasetFolder
from torchvision.transforms import ToTensor
from torch.nn.utils.rnn import pad_sequence

# Assuming your model architecture is defined in 'your_model.py'
from wf import Simple1DCNN  


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
        return torch.from_numpy(data).float(), label

def custom_collate_fn(batch):
    """Pads waveforms to the maximum length in the batch."""
    waveforms = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad waveforms
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels)

    return padded_waveforms, labels

# --- Configuration ---
checkpoint_dir = "./checkpoints"
test_data_dir = "../pdata/waveforms/testing"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Instantiate your Model ---
model = Simple1DCNN().to(device)
model.eval()

# --- Load Test Data ---
test_dataset = CustomDataset(root_dir=test_data_dir, transform=None)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# --- Function to evaluate a checkpoint ---
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            print("predicted",predicted,"where labels",labels)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# --- Iterate through checkpoints and evaluate ---
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
best_accuracy = -1
best_checkpoint = None

for checkpoint_file in checkpoint_files:
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    if os.path.basename(checkpoint_path) == "wfchkpt20.pth":
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint)
            accuracy = evaluate(model, test_loader, device)
            print(f"Checkpoint: {os.path.basename(checkpoint_path)}, Test Accuracy: {accuracy:.2f}%")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_checkpoint = os.path.basename(checkpoint_path)
        except Exception as e:
            print(f"Error loading or evaluating {os.path.basename(checkpoint_path)}: {e}")

print(f"\nBest Checkpoint: {best_checkpoint} with Test Accuracy: {best_accuracy:.2f}%")