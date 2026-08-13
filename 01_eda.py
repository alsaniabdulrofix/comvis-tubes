import pandas as pd

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("=" * 50)
print("INFORMASI DATASET")
print("=" * 50)

print(f"Jumlah data train : {len(train_df)}")
print(f"Jumlah data test  : {len(test_df)}")

print()

print("=" * 50)
print("5 DATA PERTAMA TRAIN")
print("=" * 50)

print(train_df.head())

print()

print("=" * 50)
print("JUMLAH KELAS")
print("=" * 50)

print(train_df["tag"].value_counts())

print()

print("=" * 50)
print("JUMLAH KELAS UNIK")
print("=" * 50)

print(train_df["tag"].nunique())

print()

print("=" * 50)
print("DAFTAR KELAS")
print("=" * 50)

print(sorted(train_df["tag"].unique()))