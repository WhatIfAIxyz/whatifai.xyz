import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from hypothetica_net import HypotheticaNet
from dataset import HypotheticaDataset

# Load Model
model = HypotheticaNet()
model.load_state_dict(torch.load("models/hypothetica_net.pth"))
model.eval()

# Load Tokenizer & Test Data
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
test_dataset = HypotheticaDataset("data/test_data.json", tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Evaluate Model
correct, total = 0, 0
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        outputs = model(inputs)
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"Model Accuracy: {accuracy:.2f}%")
