import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#=================Project Overview=========================#
st.title("Real Estate Price Prediction")
st.write("""
### Project Overview
This project involves predicting real estate prices based on various features such as house age, distance to the nearest MRT station, and the number of nearby convenience stores. 
We will perform data cleaning, feature engineering, scaling, and visualization to prepare the data for modeling.
""")

#=================1: Data Cleaning and Structuring=========================#
# Load the datasets
train_data_path = 'C:/Users/71591/Desktop/dataset/Train Real estate.csv'
test_data_path = 'C:/Users/71591/Desktop/dataset/Test Real estate.csv'

# Read in the datasets
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# Display the first few rows of the train and test datasets in Streamlit
st.write("### Train Data Overview")
st.write(train_df.head())

st.write("### Test Data Overview")
st.write(test_df.head())

# Convert 'X1 transaction date' into datetime and extract year and month for both train and test data
def convert_transaction_date(date):
    try:
        return pd.to_datetime(date, format='%Y.%f')
    except:
        return pd.to_datetime(date, format='%Y')

# Apply the conversion for train and test datasets
train_df['transaction_date'] = train_df['X1 transaction date'].apply(convert_transaction_date)
test_df['transaction_date'] = test_df['X1 transaction date'].apply(convert_transaction_date)

# Extract year and month from 'transaction_date'
train_df['transaction_year'] = train_df['transaction_date'].dt.year
train_df['transaction_month'] = train_df['transaction_date'].dt.month

test_df['transaction_year'] = test_df['transaction_date'].dt.year
test_df['transaction_month'] = test_df['transaction_date'].dt.month

# Drop the original 'X1 transaction date' column
train_df = train_df.drop(columns=['X1 transaction date'])
test_df = test_df.drop(columns=['X1 transaction date'])

# Display the processed train and test datasets with new date features
st.write("### Processed Train Data with Date Features")
st.write(train_df.head())

st.write("### Processed Test Data with Date Features")
st.write(test_df.head())

#=================Scaling Numerical Features=============================#
# Scale numerical features
scaler = StandardScaler()
features_to_scale = ['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']

# Fit the scaler on train data and transform both train and test data
train_df[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])
test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])

# Display the scaled data
st.write("### Scaled Train Data")
st.write(train_df.head())

st.write("### Scaled Test Data")
st.write(test_df.head())

#=================Data Visualization=========================#
st.write("### Data Visualizations")

# 1. Correlation Heatmap
st.write("#### Correlation Heatmap")
corr_matrix = train_df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
st.pyplot(plt)

# 2. Scatter Plots: Numerical features vs. Target
st.write("#### Scatter Plots")
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Scatter plot for House Age vs House Price
sns.scatterplot(x='X2 house age', y='Y house price of unit area', data=train_df, ax=ax[0])
ax[0].set_title('House Age vs House Price')

# Scatter plot for Distance to MRT vs House Price
sns.scatterplot(x='X3 distance to the nearest MRT station', y='Y house price of unit area', data=train_df, ax=ax[1])
ax[1].set_title('Distance to MRT vs House Price')

# Scatter plot for Convenience Stores vs House Price
sns.scatterplot(x='X4 number of convenience stores', y='Y house price of unit area', data=train_df, ax=ax[2])
ax[2].set_title('Number of Convenience Stores vs House Price')

st.pyplot(fig)

# 3. Histograms for Key Features
st.write("#### Histograms")
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Histogram for House Price
sns.histplot(train_df['Y house price of unit area'], bins=20, kde=True, ax=ax[0])
ax[0].set_title('House Price Distribution')

# Histogram for House Age
sns.histplot(train_df['X2 house age'], bins=20, kde=True, ax=ax[1])
ax[1].set_title('House Age Distribution')

# Histogram for Distance to MRT
sns.histplot(train_df['X3 distance to the nearest MRT station'], bins=20, kde=True, ax=ax[2])
ax[2].set_title('Distance to MRT Distribution')

st.pyplot(fig)

# 4. Box Plots: House Price by Transaction Year and Month
st.write("#### Box Plots")

# Box plot for House Price vs Transaction Year
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
sns.boxplot(x='transaction_year', y='Y house price of unit area', data=train_df, ax=ax[0])
ax[0].set_title('House Price by Transaction Year')

# Box plot for House Price vs Transaction Month
sns.boxplot(x='transaction_month', y='Y house price of unit area', data=train_df, ax=ax[1])
ax[1].set_title('House Price by Transaction Month')

st.pyplot(fig)

#=================Model Building: Linear Regression=========================#
st.write("### Model Building: Linear Regression")

# Prepare the features (X) and target (y) for training
X_train = train_df[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'transaction_year', 'transaction_month']]
y_train = train_df['Y house price of unit area']

X_test = test_df[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'transaction_year', 'transaction_month']]
y_test = test_df['Y house price of unit area']

# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test dataset
y_pred_lr = lr_model.predict(X_test)

#=================Model Evaluation=========================#
# Calculate evaluation metrics for Linear Regression
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
r2_lr = r2_score(y_test, y_pred_lr)

# Display the evaluation metrics
st.write(f"### Linear Regression Model Evaluation")
st.write(f"Mean Absolute Error (MAE): {mae_lr:.2f}")
st.write(f"Mean Squared Error (MSE): {mse_lr:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse_lr:.2f}")
st.write(f"R-squared (R2): {r2_lr:.2f}")

# Plotting Actual vs. Predicted for Linear Regression
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred_lr, alpha=0.5)
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
ax.set_title("Actual vs Predicted - Linear Regression")
ax.set_xlabel("Actual House Price")
ax.set_ylabel("Predicted House Price")
st.pyplot(fig)

#=================Model Tuning: Ridge and Lasso=========================#
st.write("### Model Tuning: Ridge and Lasso Regression")

# Initialize and train Ridge and Lasso models
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.1)

ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# Make predictions with Ridge and Lasso
y_pred_ridge = ridge_model.predict(X_test)
y_pred_lasso = lasso_model.predict(X_test)

# Evaluate Ridge
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
st.write(f"Ridge Model MAE: {mae_ridge:.2f}, R2: {r2_ridge:.2f}")

# Plotting Actual vs. Predicted for Ridge Regression
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred_ridge, alpha=0.5)
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
ax.set_title("Actual vs Predicted - Ridge Regression")
ax.set_xlabel("Actual House Price")
ax.set_ylabel("Predicted House Price")
st.pyplot(fig)

# Evaluate Lasso
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
st.write(f"Lasso Model MAE: {mae_lasso:.2f}, R2: {r2_lasso:.2f}")

# Plotting Actual vs. Predicted for Lasso Regression
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred_lasso, alpha=0.5)
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
ax.set_title("Actual vs Predicted - Lasso Regression")
ax.set_xlabel("Actual House Price")
ax.set_ylabel("Predicted House Price")
st.pyplot(fig)

#=================Additional Models: Decision Tree, Random Forest, Gradient Boosting=========================#
st.write("### Additional Models: Decision Tree, Random Forest, Gradient Boosting")

# Initialize models
dt_model = DecisionTreeRegressor(random_state=42)
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

# Train models
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Predict with the models
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

# Evaluate models (MAE and R2)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

mae_gb = mean_absolute_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# Display the evaluation metrics
st.write(f"Decision Tree MAE: {mae_dt:.2f}, R2: {r2_dt:.2f}")
st.write(f"Random Forest MAE: {mae_rf:.2f}, R2: {r2_rf:.2f}")
st.write(f"Gradient Boosting MAE: {mae_gb:.2f}, R2: {r2_gb:.2f}")

# Plotting Actual vs. Predicted for Decision Tree Regression
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred_dt, alpha=0.5)
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
ax.set_title("Actual vs Predicted - Decision Tree Regression")
ax.set_xlabel("Actual House Price")
ax.set_ylabel("Predicted House Price")
st.pyplot(fig)

# Plotting Actual vs. Predicted for Random Forest Regression
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred_rf, alpha=0.5)
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
ax.set_title("Actual vs Predicted - Random Forest Regression")
ax.set_xlabel("Actual House Price")
ax.set_ylabel("Predicted House Price")
st.pyplot(fig)

# Plotting Actual vs. Predicted for Gradient Boosting Regression
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred_gb, alpha=0.5)
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
ax.set_title("Actual vs Predicted - Gradient Boosting Regression")
ax.set_xlabel("Actual House Price")
ax.set_ylabel("Predicted House Price")
st.pyplot(fig)

#=================Cross-Validation=========================#
st.write("### Cross-Validation with Random Forest")

# Cross-validation with Random Forest
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
st.write(f"Random Forest Cross-Validation MAE: {(-rf_cv_scores.mean()):.2f}")