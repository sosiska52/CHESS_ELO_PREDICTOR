import pandas as pd
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

with open("../NeuronNet/move_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

df["moves"] = df["moves"].apply(lambda x: encoder.transform(x))

df[["white_rating", "black_rating", "moves"]].to_pickle("prepared_dataset.pkl")
print("Данные подготовлены и сохранены в prepared_dataset.pkl")

df.to_csv(output_file, index=False)
print(f"Очищенный файл сохранён как {output_file}")