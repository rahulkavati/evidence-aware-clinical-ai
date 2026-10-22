import pandas as pd

train = pd.read_parquet("data/processed/train.parquet")
val = pd.read_parquet("data/processed/val.parquet")

train_rate = train["SepsisLabel"].mean()
val_rate = val["SepsisLabel"].mean()

print(f"Training hourly positive rate: {train_rate:.4%}")
print(f"Validation hourly positive rate: {val_rate:.4%}")
print(f"Random-ranking AUPRC baseline: {val_rate:.4f}")
print(f"Constant-probability Brier baseline: {val_rate * (1-val_rate):.6f}")