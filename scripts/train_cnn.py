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

# Модель CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 111 * 111, 1000),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

# Обучение
def train():
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model.train()
    for epoch in range(5):
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/5")
        for x, y in progress_bar:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"[CNN] Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn_titanic.pt")
    print("✅ Модель CNN сохранена")

if __name__ == "__main__":
    import os
    train()
