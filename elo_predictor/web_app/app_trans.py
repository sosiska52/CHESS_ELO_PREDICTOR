import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import os


class ChessEloTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.pos_embedding = nn.Parameter(torch.zeros(1, 60, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(embed_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # усредняем по временной оси
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        return out.squeeze(1)


@st.cache_resource
def load_model_and_encoder():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "../dataset")
    model_path = os.path.join(base_dir, "../NeuronNet/elo_transformer_model.pth")

    with open(os.path.join(dataset_path, "move_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)

    with open(os.path.join(dataset_path, "rating_norm.pkl"), "rb") as f:
        norm = pickle.load(f)

    vocab_size = len(encoder.classes_)
    model = ChessEloTransformer(vocab_size=vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    return model, encoder, norm


model, encoder, norm = load_model_and_encoder()


def clean_and_encode_moves(text, encoder, max_moves=60):
    text = re.sub(r"\d+\.", "", text)  # убираем номера ходов
    text = re.sub(r"1-0|0-1|1/2-1/2|\*", "", text)  # убираем результаты
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


st.title("Chess ELO Predictor (Transformer)")
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

        st.success(f"Предполагаемый рейтинг игрока: **{prediction:.0f} ELO**")
