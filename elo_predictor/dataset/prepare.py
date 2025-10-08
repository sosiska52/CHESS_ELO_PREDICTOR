import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import itertools
import pickle

def clean_and_split_moves(text):
    moves = text.split()
    return moves

input_file = "clean_chess_games.csv"
output_file = "aug_data.csv"

df = pd.read_csv(input_file)

df["moves"] = df["moves"].apply(clean_and_split_moves)

all_moves = list(itertools.chain.from_iterable(df["moves"].tolist()))

encoder = LabelEncoder()
encoder.fit(all_moves)

with open("move_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

df["moves"] = df["moves"].apply(lambda x: encoder.transform(x))





df[["white_rating", "black_rating", "moves"]].to_pickle("prepared_dataset.pkl")
print("Данные подготовлены и сохранены в prepared_dataset.pkl")

df.to_csv(output_file, index=False)
print(f"Очищенный файл сохранён как {output_file}")



df = pd.read_pickle("prepared_dataset.pkl")

MAX_MOVES = 60

def pad_sequence(seq, max_len):
    seq = list(seq)
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        return seq + [0] * (max_len - len(seq))

df["moves"] = df["moves"].apply(lambda x: pad_sequence(x, MAX_MOVES))

X = np.vstack(df["moves"].values)
y = df["white_rating"].values

print("Форма X:", X.shape)
print("Форма y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Train:", X_train.shape, "Test:", X_test.shape)

np.savez_compressed("chess_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print("Наборы данных сохранены в 'chess_data.npz'")