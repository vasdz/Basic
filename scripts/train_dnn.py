from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Виртуальный датасет
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.rand(3, 224, 224)
        y = torch.randint(0, 1000, ())
        return x, y

# Модель DNN
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224*224*3, 1000),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

# Обучение
def train():
    model = DNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[DNN] Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    # Сохраняем модель
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/dnn_titanic.pt")
    print("Модель DNN сохранена")

if __name__ == "__main__":
    import os
    train()
