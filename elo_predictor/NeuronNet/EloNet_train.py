import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

data = np.load("../dataset/chess_data.npz")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

X_train_tensor = torch.tensor(X_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

class ChessEloBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # ×2 из-за bidirectional

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        out = self.fc(h)
        return out.squeeze(1)


import pickle

with open("../dataset/move_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

vocab_size = len(encoder.classes_)
model = ChessEloBiLSTM(vocab_size=vocab_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 200
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS}, loss: {avg_loss:.2f}")

print("Обучение завершено")

model.eval()
with torch.no_grad():
    preds = model(X_test_tensor).numpy()

mae = mean_absolute_error(y_test, preds)
rmse = math.sqrt(mean_squared_error(y_test, preds))

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("График обучения модели")
plt.show()

torch.save(model.state_dict(), "elo_lstm_model.pth")
print("✅ Модель сохранена в elo_lstm_model.pth")
