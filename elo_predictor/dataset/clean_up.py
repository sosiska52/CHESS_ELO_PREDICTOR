import pandas as pd

input_file = "games.csv"
output_file = "clean_chess_games.csv"

df = pd.read_csv(input_file)

columns_to_keep = [
    "white_rating",
    "black_rating",
    "moves"
]

df = df[columns_to_keep]

df = df.dropna()

df = df.drop_duplicates()

df = df[(df["white_rating"] >= 800) & (df["white_rating"] <= 2500)]
df = df[(df["black_rating"] >= 800) & (df["black_rating"] <= 2500)]

df = df[df["moves"].apply(lambda x: len(x.split()) > 10)]

print(f"Осталось {len(df)} партий после очистки.")

df.to_csv(output_file, index=False)
print(f"Очищенный файл сохранён как {output_file}")
