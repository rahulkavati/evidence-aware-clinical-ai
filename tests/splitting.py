import pandas as pd

train = pd.read_csv("outputs/splits/train.csv")
val = pd.read_csv("outputs/splits/val.csv")
test = pd.read_csv("outputs/splits/test.csv")

a = set(train["patient_id"])
b = set(val["patient_id"])
c = set(test["patient_id"])

assert not (a & b)
assert not (a & c)
assert not (b & c)
assert len(a | b | c) == 40336

print("PASS: No patient overlap across splits")