from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Виртуальный датасет с временной размерностью
class DummyLSTMDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=1):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.rand(self.seq_len, 3, 224, 224)  # [seq_len, C, H, W]
        y = torch.randint(0, 1000, ())
        return x, y

# Модель LSTM
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(32 * 111 * 111, 128, batch_first=True)
        self.classifier = nn.Linear(128, 1000)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.feature_extractor(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.classifier(x)

# Обучение
def train():
    model = LSTMModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = DummyLSTMDataset()
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
        print(f"[LSTM] Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    # Сохраняем модель
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_titanic.pt")
    print("Модель LSTM сохранена")

if __name__ == "__main__":
    import os
    train()
