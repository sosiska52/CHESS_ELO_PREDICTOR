import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import pickle

# ==============================
# 🔹 1. Загрузка данных
# ==============================
data = np.load("../dataset/chess_data.npz")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

# Загрузка нормализационных коэффициентов
with open("../dataset/rating_norm.pkl", "rb") as f:
    norm = pickle.load(f)

# Преобразуем в тензоры
X_train_tensor = torch.tensor(X_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# ==============================
# 🔹 2. Модель BiLSTM
# ==============================
class ChessEloBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)  # объединяем выходы двух направлений
        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        out = self.fc2(h)
        return out.squeeze(1)

# ==============================
# 🔹 3. Инициализация модели
# ==============================
with open("../dataset/move_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

vocab_size = len(encoder.classes_)
model = ChessEloBiLSTM(vocab_size=vocab_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# ==============================
# 🔹 4. Обучение модели
# ==============================
EPOCHS = 20
train_losses = []

print("🚀 Начало обучения...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")

print("\n✅ Обучение завершено!")

# ==============================
# 🔹 5. Оценка модели
# ==============================
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor).numpy()

# Денормализация прогнозов и меток
preds_real = preds * norm["std"] + norm["mean"]
y_real = y_test * norm["std"] + norm["mean"]

mae = mean_absolute_error(y_real, preds_real)
rmse = math.sqrt(mean_squared_error(y_real, preds_real))

print(f"\n📊 Результаты на тесте:")
print(f"MAE: {mae:.2f} ELO")
print(f"RMSE: {rmse:.2f} ELO")

# ==============================
# 🔹 6. Визуализация
# ==============================
plt.figure(figsize=(7, 4))
plt.plot(train_losses, label="Train Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("📉 График обучения модели")
plt.legend()
plt.show()

# ==============================
# 🔹 7. Сохранение модели
# ==============================
torch.save(model.state_dict(), "elo_lstm_model.pth")
print("💾 Модель сохранена в elo_lstm_model.pth")
