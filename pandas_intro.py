import pandas as pd
import numpy as np

# Your first DataFrame
data = {
    "name": ["Derin", "Tolu", "Kemi", "Wunmi"],
    "age": [20, 22, 21, 23],
    "score": [85, 90, 78, 88]
}

df = pd.DataFrame(data)
print(df)
print(df.shape)        # rows and columns
print(df.head())       # first 5 rows
print(df.describe())   # stats summary
print(df["score"])     # grab one column
# Filter rows where score is above 80
high_scorers = df[df["score"] > 80]
older_than_21 = df[df["age"] >= 21]
print(high_scorers)
print(older_than_21)