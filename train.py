import pandas as pd
from ml_algorithms import ml_algorithms

df = pd.read_excel("Thyroid_Diff.xlsx")

print("=" * 40 + " SOBRE O DATASET " + "=" * 40)
print(df.head())
x = input()
print("\n" + "=" * 40 + " SOBRE O DATASET " + "=" * 40)
print(df.info())
x = input()
print("\n" + "=" * 40 + " SOBRE O DATASET " + "=" * 40)
print(df.describe())
x = input()

X = df.drop("Recurred", axis=1) # features (age, gender, smoking, ...)
y = df["Recurred"] # target (recurred? -> yes/no)

X_prop = pd.get_dummies(X, drop_first=True)

train_resources = ml_algorithms()
train_resources.prepared_data(X, y)

