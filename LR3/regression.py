import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

st.title('Real Estate Price Prediction')

st.markdown("""

1. **Data Cleaning and Structuring**
2. **Exploratory Data Analysis (EDA)**
3. **Model Building**
4. **Model Evaluation**
5. **Model Deployment**
""")

train_data = pd.read_csv('Train Real estate.csv')
test_data = pd.read_csv('Test Real estate.csv')

# Rename the target column in the test data to avoid issues
if 'Y house price of unit area' in test_data.columns:
    test_data.rename(columns={'Y house price of unit area': 'Predicted price'}, inplace=True)

st.subheader('1. Data Cleaning and Structuring')

# Display the first few rows of the datasets
st.markdown("### Initial Dataset Overview")
st.write('**Training Data:**')
st.dataframe(train_data.head())
st.write('**Test Data:**')
st.dataframe(test_data.head())

# Handling missing values
st.markdown("### Handling Missing Values")
st.write("Checking for missing values in the training data:")
st.write(train_data.isnull().sum())
st.write("Checking for missing values in the test data:")
st.write(test_data.isnull().sum())

# Handling missing values - filling with mean as an example
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)
st.write("Missing values have been filled with the mean of each column.")

# Checking for duplicates
st.markdown("### Checking for Duplicates")
st.write(f"Number of duplicate rows in training data: {train_data.duplicated().sum()}")
st.write(f"Number of duplicate rows in test data: {test_data.duplicated().sum()}")

# Removing duplicates
train_data.drop_duplicates(inplace=True)
test_data.drop_duplicates(inplace=True)
st.write("Duplicates have been removed.")

# Feature engineering (if necessary)
st.markdown("### Feature Engineering")
st.write("No additional feature engineering was necessary for this dataset at this stage.")

# Dropping unneeded columns
st.markdown("### Dropping Unneeded Columns")
st.write("All columns in the provided datasets are relevant, so none were dropped.")

# Splitting the data into features and target
X_train = train_data.drop('Y house price of unit area', axis=1)
y_train = train_data['Y house price of unit area']

# Display the shape of the data
st.write(f"**Shape of X_train (features):** {X_train.shape}")
st.write(f"**Shape of y_train (target):** {y_train.shape}")

# 2. Exploratory Data Analysis (EDA)
st.subheader('2. Exploratory Data Analysis (EDA)')

# Basic statistics
st.markdown("### Basic Statistics of Features")
st.write(X_train.describe())

# Distribution of the target variable
st.markdown("### Distribution of Target Variable (House Price per Unit Area)")
plt.figure(figsize=(8, 4))
sns.histplot(y_train, kde=True)
st.pyplot(plt.gcf())  # Ensures the plot is rendered before continuing

# Pairplot to see relationships between features
st.markdown("### Pairplot to Explore Relationships Between Features")
sns.pairplot(train_data)
st.pyplot(plt.gcf())  # Ensures the plot is rendered before continuing

# Correlation heatmap
st.markdown("### Correlation Heatmap")
plt.figure(figsize=(10, 6))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot(plt.gcf())  # Ensures the plot is rendered before continuing

# Discussing insights from the heatmap
st.markdown("""
- There is a strong negative correlation between house age (`X2 house age`) and house price per unit area (`Y house price of unit area`). This suggests that newer houses tend to have higher prices.
- The distance to the nearest MRT station (`X3 distance to the nearest MRT station`) is also negatively correlated with the house price, indicating that properties closer to transit are generally more expensive.
- The number of convenience stores (`X4 number of convenience stores`) shows a moderate positive correlation with house prices, suggesting that properties with more nearby amenities are valued higher.
""")

# 3. Model Building and Comparison
st.subheader('3. Model Building and Comparison')

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Scaling the features
st.markdown("### Scaling the Features")
st.write("Feature scaling is applied to ensure that the different scales of the features don't negatively impact the regression models.")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

# Train models and evaluate them
st.markdown("### Training and Evaluating Multiple Models")

results = {}
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train_split)
    y_val_pred = model.predict(X_val_scaled)
    
    mse = mean_squared_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    
    results[model_name] = {"MSE": mse, "R2": r2}
    
    st.write(f"**{model_name}:**")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-squared (R2): {r2:.2f}")
    st.write("---")

# Comparing models
st.subheader('4. Model Comparison')

# Results DataFrame
results_df = pd.DataFrame(results).T
st.write("### Comparison of Model Performance")
st.dataframe(results_df)


# 5. Conclusion and Model Deployment
st.subheader('5. Conclusion and Model Deployment')

# Best model selection
best_model_name = results_df['R2'].idxmax()
best_model = models[best_model_name]

st.write(f"**The best performing model is: {best_model_name}** with an R-squared value of {results_df.loc[best_model_name, 'R2']:.2f}")

# Drop the target column from the test dataset if it exists
if 'Predicted price' in test_data.columns:
    test_features = test_data.drop('Predicted price', axis=1)
else:
    test_features = test_data

# Predicting on test data with the best model
test_data_scaled = scaler.transform(test_features)
test_predictions = best_model.predict(test_data_scaled)

st.markdown("### Predictions on Test Data with Best Model")
st.write(test_predictions)

st.write("# PLEASE REFRESH THE PAGE IF SOME PLOTS DO NOT SHOW.")
