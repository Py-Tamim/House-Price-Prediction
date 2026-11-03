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
df["Region"] = df["Region"].astype(str).str.strip()
df["Sub-region"] = df["Sub-region"].fillna("Unknown").astype(str).str.strip()

# Display a short overview of the loaded dataset.
print("\n========== DATASET OVERVIEW ==========")
print(f"Total usable listings: {len(df)}")
print("\nFirst five listings:")
print(df.head())

region_encoder = LabelEncoder()
df["Region_enc"] = region_encoder.fit_transform(df["Region"])


X_price = df[["Price"]]                             # Train The ML Modles
model_region = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model_region.fit(X_price, df["Region_enc"])
model_beds = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model_beds.fit(X_price, df["No. Beds"])
model_baths = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model_baths.fit(X_price, df["No. Baths"])
X_area = df[["Area"]]

model_price = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model_price.fit(X_area, df["Price"])
print("\nMachine learning models trained successfully.")

def plot_listings(result, title):                    # Function For Ploting Graphs.
    if result.empty:
        print("No results available to plot.")
        return
    result = result.copy()
    result = result.head(10)
    plt.figure(figsize=(10, 6))

    labels = (
        result["Region"].astype(str)
        + " - "
        + result["Sub-region"].astype(str)
    )
    plt.bar(labels, result["Price"])
    plt.title(title)
    plt.xlabel("Region and Sub-region")
    plt.ylabel("Price (Taka)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
#================================= Continue =================================