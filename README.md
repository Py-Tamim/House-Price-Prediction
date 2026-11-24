# House Price Prediction System Using Machine Learning 🏠

A Python-based Machine Learning project designed to analyze residential property listings and predict house prices based on property area, location, and historical pricing data.

The system uses **Random Forest Classification and Regression** to analyze property information and provide house price predictions, property searches, and price-based recommendations.

Users can search for houses by price or region, estimate property prices by entering the house area, and explore low-, mid-, and high-priced properties across different regions.

## 📌 Project Overview

Finding a suitable property within a specific budget can be challenging, especially in a competitive real estate market like Bangladesh.

This project aims to explore how machine learning can help users understand property pricing patterns and make more informed decisions when searching for residential properties.

Using historical property listing data, the system provides three main functionalities:

1. **House Search:** Find properties using a specific price or region.
2. **Price Prediction:** Estimate the price of a property based on its area in square feet.
3. **Price-Based Recommendations:** Explore low-, mid-, and high-priced properties, with optional region filtering.

The system also generates bar charts to compare property prices and visualize estimated house values.

## ✨ Key Features

- **House Search by Price:** Finds properties within a specified price range.
- **Region-Based Search:** Displays available property listings for a selected region.
- **House Price Prediction:** Estimates property prices based on house area.
- **Price Range Estimation:** Provides an illustrative price range around the predicted value.
- **Low, Mid & High Price Filtering:** Categorizes properties based on the dataset's price distribution.
- **Location-Based Filtering:** Filters price categories by region, such as Mirpur.
- **Property Comparison:** Displays property information, including location, price, bedrooms, bathrooms, and area.
- **Data Visualization:** Generates bar charts to compare property prices and prediction results.
- **Interactive Terminal Menu:** Allows users to access different features through a simple command-line interface.

## 🤖 Machine Learning Models

This project uses two Random Forest algorithms for different machine learning tasks.

### 1. Random Forest Classifier

The classification models are used to identify property characteristics based on the listed house price.

Three separate classifiers are trained:

| Model | Input Feature | Predicted Output |
|---|---|---|
| Region Classifier | Price | Region |
| Bedroom Classifier | Price | Number of Beds |
| Bathroom Classifier | Price | Number of Baths |

**Model Configuration:**

- Algorithm: `RandomForestClassifier`
- Number of estimators: 100
- Random state: 42

These models learn relationships between property prices and their corresponding categorical values.

### 2. Random Forest Regressor

The regression model estimates house prices based on property area.

**Input Feature:**
- Area (square feet)

**Predicted Output:**
- Estimated House Price (BDT)

**Model Configuration:**

- Algorithm: `RandomForestRegressor`
- Number of estimators: 100
- Random state: 42

The model learns from historical property listings to estimate the relationship between house area and price.

### Price Range Estimation

After generating a predicted house price, the system calculates an illustrative range using a 15% margin.

The calculation is:

\[
\text{Lower Price} = \text{Predicted Price} \times 0.85
\]

\[
\text{Upper Price} = \text{Predicted Price} \times 1.15
\]

This produces an estimated price range of ±15% around the model's prediction.

**Note:** This range is an illustrative margin, not a statistically calibrated prediction interval.

## 📂 Dataset Information

**Dataset Name:** `bproperty_data for ML projct.csv`

The project uses property listing information containing the following columns:

| Column | Description |
|---|---|
| Region | Main location or region of the property |
| Sub-region | Specific area or neighborhood |
| Price | Listed property price in Bangladeshi Taka |
| No. Beds | Number of bedrooms |
| No. Baths | Number of bathrooms |
| Area | Property area in square feet |

### Data Preprocessing

Before training the machine learning models, the dataset goes through several cleaning operations:

- Remove unnecessary spaces from column names.
- Convert Price, No. Beds, No. Baths, and Area into numeric formats.
- Remove records with missing values in required columns.
- Remove properties with zero or negative prices.
- Remove properties with zero or negative areas.
- Remove records with missing or empty region names.
- Replace missing Sub-region values with `Unknown`.
- Convert Region and Sub-region values into string format.
- Encode region names into numerical values using `LabelEncoder`.

These preprocessing steps help prepare the property dataset for machine learning.

## 🔄 Project Workflow

### 1. Data Loading and Cleaning

The CSV dataset is loaded using Pandas.

The system checks whether all required columns are available and removes invalid property records before model training.

### 2. Feature Preparation

The dataset is divided into input features and target variables depending on the machine learning task.

For classification, the property price is used to predict region, bedroom count, and bathroom count.

For regression, property area is used to estimate the house price.

### 3. Model Training

The Random Forest classification and regression models are trained using the cleaned dataset.

Each model learns from historical property listing patterns.

### 4. User Input and Prediction

Users can interact with the trained models through a terminal-based menu.

Depending on the selected option, the system searches existing property listings or generates a new price estimate.

### 5. Data Visualization

Matplotlib is used to generate visual representations of property prices and estimated values.

This helps users understand the differences between available properties and compare price information.

## 🖥️ System Functionalities

The program provides three main operations through an interactive menu.

### Option 1: Search Houses by Price or Region

Users can enter either a property price or a region name.

**Example: Search by Price**

```text
Enter a price in Taka or a region name:
5000000
```

The system searches for properties within Tk 100,000 above or below the entered price and displays the closest matching listings.

**Example: Search by Region**

```text
Enter a price in Taka or a region name:
Mirpur
```

The system displays available properties in Mirpur, including their prices, bedrooms, bathrooms, and areas.

### Option 2: Predict House Price from Area

Users can enter the property area in square feet to estimate its price.

**Example Input:**

```text
Enter house area in sq ft:
1500
```

**Example Output:**

```text
========== PRICE PREDICTION ==========

House area: 1,500 sq ft
Estimated price: Tk 7,500,000
Estimated range: 63.75 to 86.25 Lakh Taka
```

*Note: The values above are illustrative examples, not actual model output.*

The system also generates a bar chart showing the predicted house price and its illustrative ±15% range.

### Option 3: Search Low-, Mid-, or High-Price Houses

Users can describe the type of property they want.

**Example Inputs:**

```text
low price house
```

```text
mid price house in Mirpur
```

```text
high price house
```

The system divides property prices into three categories using the 33rd and 66th percentiles of the dataset.

| Category | Price Condition |
|---|---|
| Low | Price at or below the 33rd percentile |
| Mid | Price between the 33rd and 66th percentiles |
| High | Price at or above the 66th percentile |

Users can also include a region name to narrow down the search results.

The system displays matching property listings and generates a bar chart for comparison.

## 📈 Data Visualization

The project uses Matplotlib to generate two types of visualizations.

### 1. Property Price Comparison

A bar chart displays the prices of up to ten matching properties.

The chart includes:

- Region and sub-region labels.
- Property prices in Bangladeshi Taka.
- Rotated location labels for readability.
- Grid lines for easier price comparison.

### 2. Predicted House Price

A separate bar chart displays the estimated price for a user-provided property area.

It also shows the illustrative ±15% price range using error bars.

These visualizations make property price comparisons and prediction results easier to understand.

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading, cleaning, and manipulation |
| Scikit-learn | Machine learning algorithms |
| RandomForestClassifier | Property classification |
| RandomForestRegressor | House price prediction |
| LabelEncoder | Encoding categorical region values |
| Matplotlib | Data visualization |

## 📦 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### 2. Navigate to the Project Directory

```bash
cd YOUR_REPOSITORY
```

### 3. Install Required Libraries

```bash
pip install pandas scikit-learn matplotlib
```

Alternatively, if a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Place the CSV dataset in the same directory as the Python script.

```text
House-Price-Prediction/
│
├── house_price_prediction.py
├── bproperty_data for ML projct.csv
├── README.md
└── requirements.txt
```

### 5. Run the Program

```bash
python house_price_prediction.py
```

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── house_price_prediction.py
├── bproperty_data for ML projct.csv
├── README.md
└── requirements.txt
```

**File Description:**

- `house_price_prediction.py` – Main Python script containing data preprocessing, model training, house searching, price prediction, and visualization.
- `bproperty_data for ML projct.csv` – Property listing dataset used for training and searching.
- `README.md` – Project documentation.
- `requirements.txt` – Required Python libraries.

## ⚠️ Limitations

Although the system provides a machine learning-based approach to property price analysis, it has some limitations:

- Predictions depend on the quality and quantity of historical property listings.
- The current price prediction model uses only property area as its input feature.
- Location, bedroom count, bathroom count, building age, and other property characteristics are not included in the regression model.
- The classification models are trained separately and are not currently integrated into the house price prediction process.
- The ±15% estimated price range is a manually selected margin, not a model-generated confidence interval.
- The model does not consider current real estate market conditions or future price changes.
- Predictions should be treated as estimates rather than guaranteed property market values.

## 🚀 Future Improvements

Some possible improvements for this project include:

- Including location, bedroom count, and bathroom count in the price prediction model.
- Comparing Random Forest with Linear Regression, Decision Tree, XGBoost, and other regression algorithms.
- Improving prediction accuracy through feature engineering and hyperparameter tuning.
- Adding property size and location-based price comparisons.
- Developing an interactive dashboard using Streamlit.
- Integrating a property recommendation system based on user budgets and preferences.
- Adding geographical maps to visualize property locations.
- Evaluating model performance using MAE, RMSE, and R² Score.

## 🤝 AI Assistance & Acknowledgement

During the development of this project, I used AI tools as a supporting resource for research, planning, and technical guidance.

AI assistance was mainly used for:

- Exploring and refining the initial house price prediction project idea.
- Finding and understanding suitable CSV datasets containing residential property information.
- Comparing different machine learning algorithms and understanding which models would be suitable for property classification and price prediction.
- Understanding the use of Random Forest Classification and Regression for the project.
- Exploring data preprocessing techniques and possible improvements to the project workflow.

I used AI as a learning and research assistant to explore different approaches and improve my understanding of the concepts involved.


The project implementation, data preprocessing, model training, testing, and result analysis were part of my own development and learning process. I reviewed the suggestions and applied them based on the requirements of the project.
## 📝 Conclusion

This project demonstrates how machine learning can be applied to residential property data to analyze housing prices and support property searches.
By combining Random Forest Classification, Random Forest Regression, data preprocessing, and visualization, the system provides a simple way to explore property listings and estimate house prices based on available historical data.
The project also provided practical experience with Python programming, data analysis, machine learning algorithms, and real-world dataset handling.
## 📄 License

This project is intended for educational and research purposes. You may explore, modify, and extend it with appropriate attribution.
