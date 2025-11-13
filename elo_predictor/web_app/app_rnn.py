import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import os


class ChessEloRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, num_layers=2, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.rnn(x)
        h = h[-1]
        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        out = self.fc2(h)
        return out.squeeze(1)


@st.cache_resource
def load_model_and_encoder():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "../dataset")
    model_path = os.path.join(base_dir, "../NeuronNet/elo_rnn_model.pth")

    with open(os.path.join(dataset_path, "move_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)

    with open(os.path.join(dataset_path, "rating_norm.pkl"), "rb") as f:
        norm = pickle.load(f)

    vocab_size = len(encoder.classes_)
    model = ChessEloRNN(vocab_size=vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    return model, encoder, norm



model, encoder, norm = load_model_and_encoder()


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
    while len(encoded) < max_moves:
        encoded.append(0)
    return torch.tensor([encoded], dtype=torch.long)


st.title("Chess ELO Predictor (RNN)")
st.write("Введите партию в стандартной шахматной нотации (например, `e4 e5 Nf3 Nc6 Bb5 a6 ...`).")

user_input = st.text_area("Ходы:", height=150)

if st.button("Предсказать рейтинг"):
    if len(user_input.strip()) == 0:
        st.warning("Введите хотя бы один ход.")
    else:
        encoded_moves = clean_and_encode_moves(user_input, encoder)
        with torch.no_grad():
            pred_norm = model(encoded_moves).item()

        prediction = pred_norm * norm["std"] + norm["mean"]

        st.success(f"Предполагаемый рейтинг игрока: {prediction:.0f} ELO")
