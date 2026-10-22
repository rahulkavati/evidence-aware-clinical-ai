import pandas as pd

train = pd.read_parquet("data/processed/train.parquet")
val = pd.read_parquet("data/processed/val.parquet")
test = pd.read_parquet("data/processed/test.parquet")

assert set(train.patient_id).isdisjoint(set(val.patient_id))
assert set(train.patient_id).isdisjoint(set(test.patient_id))
assert set(val.patient_id).isdisjoint(set(test.patient_id))

assert len(train) + len(val) + len(test) == 1552210

assert "SepsisLabel" in train.columns

print("PASS: Dataset integrity checks")
print("Train:", train.shape)
print("Validation:", val.shape)
print("Test:", test.shape)