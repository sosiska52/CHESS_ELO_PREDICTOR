import streamlit as st
import torch
import torch.nn as nn
import pickle
import re

# ====== 1. Определяем архитектуру модели ======
class ChessEloBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 из-за двунаправленности

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)  # объединяем прямое и обратное состояния
        h = self.dropout(h)
        out = self.fc(h)
        return out.squeeze(1)


# ====== 2. Загрузка модели и энкодера ======
@st.cache_resource
def load_model_and_encoder():
    with open("../dataset/move_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    vocab_size = len(encoder.classes_)
    model = ChessEloBiLSTM(vocab_size=vocab_size)

    model.load_state_dict(torch.load("../NeuronNet/elo_lstm_model.pth", map_location=torch.device("cpu")))
    model.eval()

    return model, encoder

model, encoder = load_model_and_encoder()

# ====== 3. Вспомогательная функция ======
def clean_and_encode_moves(text, encoder, max_moves=60):
    text = re.sub(r"\d+\.", "", text)
    text = re.sub(r"1-0|0-1|1/2-1/2|\*", "", text)
    text = text.strip()
    moves = text.split()
    moves = moves[:max_moves]
    encoded = []
    for m in moves:
        if m in encoder.classes_:
            encoded.append(encoder.transform([m])[0])
        else:
            encoded.append(0)
    # padding
    while len(encoded) < max_moves:
        encoded.append(0)
    return torch.tensor([encoded], dtype=torch.long)

# ====== 4. Интерфейс ======
st.title("♟️ Chess ELO Predictor")
st.write("Введите партию в стандартной шахматной нотации (например, `e4 e5 Nf3 Nc6 Bb5 a6 ...`)")

user_input = st.text_area("Ходы:", height=150)
if st.button("🔮 Предсказать рейтинг"):
    if len(user_input.strip()) == 0:
        st.warning("Введите хотя бы один ход.")
    else:
        encoded_moves = clean_and_encode_moves(user_input, encoder)
        with torch.no_grad():
            prediction = model(encoded_moves).item()
        st.success(f"Предполагаемый рейтинг игрока: **{prediction:.0f} ELO**")
