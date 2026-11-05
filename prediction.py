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

    def search_house(user_input):                    # User Input Function (price or region name).
        user_input = user_input.strip()
        try:
            price = float(user_input)

        except ValueError:
            result = df[
                df["Region"].str.casefold() == user_input.casefold()
            ]
            if result.empty:
                print("\nNo listings found for this region.")
                return
            columns = [
                "Region", "Sub-region", "Price",
                "No. Beds", "No. Baths", "Area"
            ]

            print("\n========== REGION SEARCH RESULTS ==========")
            print(result[columns].head(10).to_string(index=False))
            plot_listings(result, f"Houses in {user_input}")
            return

        if price <= 0:
            print("Please enter a price greater than zero.")
            return
        tolerance = 100000
        result = df[
            (df["Price"] >= price - tolerance)
            & (df["Price"] <= price + tolerance)
        ].copy()

        if result.empty:
            print("\nNo listings found near this price.")
            return
        result["Price_Difference"] = (
            result["Price"] - price
        ).abs()

        result = result.sort_values("Price_Difference")
        columns = [
            "Region", "Sub-region", "Price",
            "No. Beds", "No. Baths", "Area"
        ]
        print("\n========== PRICE SEARCH RESULTS ==========")
        print(result[columns].head(10).to_string(index=False))
        plot_listings(result, f"Houses Near Tk {price:,.0f}")

def predict_from_area(area):

    if area <= 0:
        print("Area must be greater than zero.")
        return
    input_data = pd.DataFrame(
        {"Area": [area]}
    )

    predicted_price = model_price.predict(input_data)[0]
    low_price = predicted_price * 0.85
    high_price = predicted_price * 1.15
    low_lakh = low_price / 100000
    high_lakh = high_price / 100000

    print("\n========== PRICE PREDICTION ==========")
    print(f"House area: {area:,.0f} sq ft")
    print(f"Estimated price: Tk {predicted_price:,.0f}")
    print(
        f"Estimated range: {low_lakh:.2f} to "
        f"{high_lakh:.2f} Lakh Taka"
    )
    plt.figure(figsize=(8, 5))

    # Draw a bar showing the central predicted price.
    plt.bar(
        ["Predicted Price"],
        [predicted_price],
        label="Predicted price"
    )
    plt.errorbar(
        ["Predicted Price"],
        [predicted_price],
        yerr=[[
            predicted_price - low_price
        ], [
            high_price - predicted_price
        ]],
        fmt="none",
        capsize=10,
        label="Illustrative +/-15% range"
    )
    plt.title(f"Predicted Price for a {area:,.0f} sq ft House")
    plt.ylabel("Price (Taka)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


low_cut = df["Price"].quantile(0.33)              # Low, mid & high price search
high_cut = df["Price"].quantile(0.66)

def search_by_text(user_text):
    text = user_text.casefold()
    if "low" in text:
        result = df[df["Price"] <= low_cut].copy()
        category = "Low-Price Houses"
    elif "high" in text:
        result = df[df["Price"] >= high_cut].copy()
        category = "High-Price Houses"
    elif "mid" in text or "medium" in text:
        result = df[
            (df["Price"] > low_cut)
            & (df["Price"] < high_cut)
        ].copy()
        category = "Mid-Price Houses"
    else:
        print(
            "Please mention low, mid, medium, or high "
            "in your request."
        )
        return
    region_filter = None
    for region in df["Region"].unique():
        if region.casefold() in text:
            region_filter = region
            break
    if region_filter:
        result = result[
            result["Region"].str.casefold()
            == region_filter.casefold()
        ]
    result = result.drop_duplicates(
        subset=["Region", "Sub-region"]
    )
    result = result.sort_values("Price")
    columns = [
        "Region", "Sub-region", "Price",
        "No. Beds", "No. Baths", "Area"
    ]
    print(f"\n========== {category.upper()} ==========")
    print(f"Low-price threshold: Tk {low_cut:,.0f}")
    print(f"High-price threshold: Tk {high_cut:,.0f}")
    if region_filter:
        print(f"Region filter: {region_filter}")
    if result.empty:
        print("No matching houses were found.")
        return
    print(result[columns].head(10).to_string(index=False))
    plot_listings(result, category)

def main():
    while True:                                 # Continious Loop of output
        print("\n==========================================")
        print("     HOUSE PRICE PREDICTION SYSTEM")
        print("==========================================")
        print("1. Search houses by price or region")
        print("2. Predict price from area")
        print("3. Search low-, mid-, or high-price houses")
        print("4. Exit")
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            user_input = input(
                "Enter a price in Taka or a region name: "
            )
            search_house(user_input)
        elif choice == "2":
            try:
                area = float(
                    input("Enter house area in sq ft: ")
                )
                predict_from_area(area)

            except ValueError:                    # raise value error if an invalid area is entered.
                print("Please enter a valid numeric area.")
        elif choice == "3":
            user_text = input(
                "Describe the house you want "
                "(e.g., low price house in Mirpur): "
            )
            search_by_text(user_text)
        elif choice == "4":
            print("Thank you for using the House Price Prediction System.")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":             # Run the main function if this script is executed directly.
    main()
