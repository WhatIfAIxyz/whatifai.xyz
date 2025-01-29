import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from hypothetica_net import HypotheticaNet
from dataset import HypotheticaDataset

# Load Model & Tokenizer
model = HypotheticaNet()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load Data
train_dataset = HypotheticaDataset("data/train_data.json", tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Optimizer & Loss Function
optimizer = optim.Adam(model.parameters(), lr=3e-5)
criterion = torch.nn.BCELoss()

# Training Loop
for epoch in range(3):
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed, Loss: {loss.item():.4f}")

# Save Model
torch.save(model.state_dict(), "models/hypothetica_net.pth")
print("Model training complete & saved.")
