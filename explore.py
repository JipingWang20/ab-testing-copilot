import pandas as pd

df = pd.read_csv("data/cookie_cats.csv")

print("Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nGroup sizes:")
print(df["version"].value_counts())
print("\nRetention means by group:")
print(df.groupby("version")[["retention_1", "retention_7"]].mean())
