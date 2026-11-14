"""Lab 3: NumPy, Pandas, preprocessing, EDA and visualization."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Small sample data; replace this with pd.read_csv("your_file.csv") for your dataset.
df = pd.DataFrame({
    "Name": ["Asha", "Bharat", "Chetan", "Diya", "Eshan", "Farah"],
    "Age": [20, 21, np.nan, 22, 20, 21],
    "Marks": [78, 85, 69, np.nan, 92, 85],
    "Department": ["IT", "CS", "IT", "CS", "IT", "CS"]
})

print("NumPy array:", np.array([10, 20, 30]))
print("\nFirst rows:\n", df.head())
print("\nShape:", df.shape)
print("\nInfo:"); df.info()
print("\nStatistics:\n", df.describe(include="all"))
print("\nMissing values:\n", df.isnull().sum())

# Fill numeric missing values with column mean.
for col in ["Age", "Marks"]:
    df[col] = df[col].fillna(df[col].mean())

# Encode categorical column.
encoder = LabelEncoder()
df["Department_Code"] = encoder.fit_transform(df["Department"])

# Normalize numeric columns to range [0, 1].
scaler = MinMaxScaler()
df[["Age_Scaled", "Marks_Scaled"]] = scaler.fit_transform(df[["Age", "Marks"]])

print("\nCleaned data:\n", df)
print("\nCorrelation:\n", df[["Age", "Marks"]].corr())

df["Marks"].plot(kind="bar", title="Student Marks", xlabel="Student index", ylabel="Marks")
plt.tight_layout()
plt.savefig("lab3_marks_bar_chart.png")
plt.show()
