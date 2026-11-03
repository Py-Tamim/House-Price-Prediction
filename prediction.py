import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("bproperty_data for ML projct.csv")             #Read the CSV file.
df.columns = df.columns.str.strip()
required_columns = [
    "Region", "Sub-region", "Price",
    "No. Beds", "No. Baths", "Area"
]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]
if missing_columns:
    raise ValueError(
        f"Missing required columns in CSV: {missing_columns}"
    )
for column in ["Price", "No. Beds", "No. Baths", "Area"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(                                             # Drop or remove rows with missing values in the specified columns.
    subset=[
        "Region", "Price", "No. Beds",
        "No. Baths", "Area"
    ]
).copy()
df = df[(df["Price"] > 0) & (df["Area"] > 0)].copy()
df = df[
    df["Region"].notna()
    & (df["Region"].astype(str).str.strip() != "")
].copy()

if df.empty:
    raise ValueError("The CSV contains no usable house records.")