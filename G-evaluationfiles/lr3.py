import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Set up Seaborn styles for colorful graphs
sns.set(style="whitegrid", palette="muted", color_codes=True)

# Title of the Streamlit app
st.title('Energy Consumption Forecasting')
st.write("""
In this app, we aim to clean the dataset, perform exploratory data analysis, and build a regression model to predict energy consumption. 
Effective energy forecasting is crucial for optimizing energy usage and implementing efficient strategies.
""")

# Step 1: Data Collection, Connectivity, and Cleaning
st.header("1. Data Collection and Cleaning")
st.write("""
We begin by loading the dataset, checking for any data issues (like missing values or duplicates), and cleaning the data. 
Proper data cleaning is crucial for building accurate models, as poor data quality can lead to misleading results.
""")

# Load the dataset
df = pd.read_excel("C:/Users/71589/Data-Science-Evaluation/LR2/Train_ENB2012_data.xlsx")

# Rename columns to more descriptive names
df.columns = ['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area',
              'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution',
              'Heating Load', 'Cooling Load']

# Data Cleaning
st.subheader("Data Cleaning Overview")
st.write("""
We first check for missing values and duplicate rows, which can negatively affect our model's performance if left untreated. 
We then impute any missing values and remove any duplicates to ensure a clean dataset.
""")

# Check for missing values and duplicate rows
missing_values = df.isnull().sum()
duplicate_rows = df.duplicated().sum()

# Display missing values and duplicate rows
st.write("Missing values in each column:")
st.write(missing_values)
st.write(f"Duplicate rows in the dataset: {duplicate_rows}")

# Ensure data types are correct
df['Orientation'] = df['Orientation'].astype('float64')
df['Glazing Area Distribution'] = df['Glazing Area Distribution'].astype('float64')

# Handle missing values by imputing with the mean
df = df.fillna(df.mean())

# Display basic information
st.subheader("Dataset Overview")
st.write("""
Here is a glimpse of the dataset after handling missing values. The table below shows the first few rows of the dataset, and the summary statistics give you an idea of the distribution of each feature.
""")
st.write(df.head())
st.write(df.describe())

# Outlier Detection using Z-scores
st.subheader("Outlier Detection and Handling")
st.write("""
Outliers can skew our model's predictions and lead to less accurate results. 
In this section, we use Z-scores to detect and visualize outliers in the dataset. We then offer the option to remove these outliers for a cleaner dataset.
""")

# Statistical Outlier Detection using Z-scores
z_scores = np.abs(zscore(df.drop(['Heating Load', 'Cooling Load'], axis=1)))
outliers = np.where(z_scores > 3)
outlier_count = len(outliers[0])
st.write(f"Detected {outlier_count} outliers in the dataset using Z-scores.")

# Visualize outliers using box plots before removal
st.subheader("Visualizing Outliers Before Removal")
st.write("""
Below are the boxplots of each feature in the dataset. Outliers, if present, are displayed as individual points outside the 'whiskers' of the boxplot.
""")

fig, ax = plt.subplots(4, 2, figsize=(15, 20))

sns.boxplot(y=df['Relative Compactness'], ax=ax[0, 0])
ax[0, 0].set_title('Boxplot of Relative Compactness Before Removal', fontsize=14)

sns.boxplot(y=df['Surface Area'], ax=ax[0, 1])
ax[0, 1].set_title('Boxplot of Surface Area Before Removal', fontsize=14)

sns.boxplot(y=df['Wall Area'], ax=ax[1, 0])
ax[1, 0].set_title('Boxplot of Wall Area Before Removal', fontsize=14)

sns.boxplot(y=df['Roof Area'], ax=ax[1, 1])
ax[1, 1].set_title('Boxplot of Roof Area Before Removal', fontsize=14)

sns.boxplot(y=df['Overall Height'], ax=ax[2, 0])
ax[2, 0].set_title('Boxplot of Overall Height Before Removal', fontsize=14)

sns.boxplot(y=df['Orientation'], ax=ax[2, 1])
ax[2, 1].set_title('Boxplot of Orientation Before Removal', fontsize=14)

sns.boxplot(y=df['Glazing Area'], ax=ax[3, 0])
ax[3, 0].set_title('Boxplot of Glazing Area Before Removal', fontsize=14)

sns.boxplot(y=df['Glazing Area Distribution'], ax=ax[3, 1])
ax[3, 1].set_title('Boxplot of Glazing Area Distribution Before Removal', fontsize=14)

plt.tight_layout()
st.pyplot(fig)

# Optionally remove outliers
df_cleaned = df[(z_scores < 3).all(axis=1)]
st.write(f"Outliers removed. {df_cleaned.shape[0]} rows remaining.")

# Visualize outliers using box plots after removal
st.subheader("Visualizing Outliers After Removal")
st.write("""
After removing the outliers, let's take another look at the boxplots to ensure that the outliers have been handled properly.
""")

fig, ax = plt.subplots(4, 2, figsize=(15, 20))

sns.boxplot(y=df_cleaned['Relative Compactness'], ax=ax[0, 0])
ax[0, 0].set_title('Boxplot of Relative Compactness After Removal', fontsize=14)

sns.boxplot(y=df_cleaned['Surface Area'], ax=ax[0, 1])
ax[0, 1].set_title('Boxplot of Surface Area After Removal', fontsize=14)

sns.boxplot(y=df_cleaned['Wall Area'], ax=ax[1, 0])
ax[1, 0].set_title('Boxplot of Wall Area After Removal', fontsize=14)

sns.boxplot(y=df_cleaned['Roof Area'], ax=ax[1, 1])
ax[1, 1].set_title('Boxplot of Roof Area After Removal', fontsize=14)

sns.boxplot(y=df_cleaned['Overall Height'], ax=ax[2, 0])
ax[2, 0].set_title('Boxplot of Overall Height After Removal', fontsize=14)

sns.boxplot(y=df_cleaned['Orientation'], ax=ax[2, 1])
ax[2, 1].set_title('Boxplot of Orientation After Removal', fontsize=14)

sns.boxplot(y=df_cleaned['Glazing Area'], ax=ax[3, 0])
ax[3, 0].set_title('Boxplot of Glazing Area After Removal', fontsize=14)

sns.boxplot(y=df_cleaned['Glazing Area Distribution'], ax=ax[3, 1])
ax[3, 1].set_title('Boxplot of Glazing Area Distribution After Removal', fontsize=14)

plt.tight_layout()
st.pyplot(fig)

# Re-check and handle any new NaN values that may have appeared after outlier removal
df_cleaned = df_cleaned.fillna(df_cleaned.mean())

# Inconsistency Checks
st.subheader("Inconsistency Checks")
st.write("""
Next, we check for any inconsistencies in the data. 
These include values that are outside of the expected range for each feature, which can be an indicator of data entry errors or other issues.
""")

# Define expected ranges for each feature
expected_ranges = {
    'Relative Compactness': (0.5, 1.0),
    'Surface Area': (400.0, 900.0),
    'Wall Area': (200.0, 450.0),
    'Roof Area': (100.0, 300.0),
    'Overall Height': (2.5, 10.0),
    'Orientation': (1, 4),
    'Glazing Area': (0.0, 0.4),
    'Glazing Area Distribution': (0, 5)
}

# Check for inconsistencies
inconsistent_data = False
for feature, (min_val, max_val) in expected_ranges.items():
    if not df_cleaned[feature].between(min_val, max_val).all():
        inconsistent_data = True
        st.write(f"Inconsistencies found in {feature}:")
        st.write(df_cleaned[~df_cleaned[feature].between(min_val, max_val)])

if not inconsistent_data:
    st.write("No inconsistencies found in the data.")

# Continue with the cleaned and checked dataset
df = df_cleaned

# Display the cleaned dataset
st.subheader("Cleaned Dataset Overview")
st.write(df.head())
st.write(df.describe())

# Step 2: Exploratory Data Analysis (EDA)
st.header("2. Exploratory Data Analysis (EDA)")
st.write("""
In this section, we will ill explore the relationships between the features and the target variables (Heating Load and Cooling Load).
This exploration helps us understand how different factors impact energy consumption, guiding us in selecting features for the model.
""")

# Correlation Analysis
st.subheader("Correlation Analysis")
st.write("""
We start by analyzing the correlation between different features and the target variables.
Understanding these correlations helps us identify which features are strongly related to energy consumption and can be useful for prediction.
""")

# Display the correlation matrix as a heatmap
correlation_matrix = df.corr().round(2)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of Features')
st.pyplot(plt.gcf())

st.write("""
From the heatmap above, we can identify features that have strong correlations with the target variables. 
High correlations between features and target variables indicate that those features are likely good predictors of energy consumption.
""")

# Pairplot for visualizing relationships
st.subheader("Pairplot Analysis")
st.write("""
To further visualize the relationships between features and target variables, we use pair plots. 
Pair plots show scatter plots of feature pairs, giving us an idea of linear relationships and potential interactions.
""")

sns.pairplot(df, diag_kind="kde", plot_kws={'alpha':0.6, 's':80, 'edgecolor':'k'}, markers='o')
plt.suptitle('Pairplot of Features and Target Variables', y=1.02)
st.pyplot(plt.gcf())

# Analyze specific features
st.subheader("Feature Analysis")
st.write("""
Let's take a closer look at some specific features to see how they relate to the target variables.
""")

# Example: Analyzing the impact of 'Relative Compactness' on 'Heating Load' and 'Cooling Load'
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.scatterplot(data=df, x='Relative Compactness', y='Heating Load', ax=ax[0])
ax[0].set_title('Heating Load vs. Relative Compactness')

sns.scatterplot(data=df, x='Relative Compactness', y='Cooling Load', ax=ax[1])
ax[1].set_title('Cooling Load vs. Relative Compactness')

plt.tight_layout()
st.pyplot(fig)

st.write("""
We can see from the scatter plots that 'Relative Compactness' has a noticeable impact on both 'Heating Load' and 'Cooling Load'.
This feature seems to be a strong predictor and should be retained for model building.
""")

# Step 3: Regression Model Building
st.header("3. Regression Model Building")
st.write("""
Now that we've cleaned the data and explored the relationships between features, it's time to build our regression model to predict energy consumption.
We will start with a simple Linear Regression model using the cleaned features.
""")

# Feature Scaling
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop(['Heating Load', 'Cooling Load'], axis=1)), columns=df.columns[:-2])

# Define Features and Target Variables
X = df_scaled  # Features (excluding Heating Load and Cooling Load)
Y = df[['Heating Load', 'Cooling Load']]  # Target variables

# Split the Data (for training and validation)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict on the test set for both Heating Load and Cooling Load
Y_test_pred = model.predict(X_test)

# Evaluate the Model for Heating Load
mae_y1 = mean_absolute_error(Y_test['Heating Load'], Y_test_pred[:, 0])
mse_y1 = mean_squared_error(Y_test['Heating Load'], Y_test_pred[:, 0])
r2_y1 = r2_score(Y_test['Heating Load'], Y_test_pred[:, 0])

# Evaluate the Model for Cooling Load
mae_y2 = mean_absolute_error(Y_test['Cooling Load'], Y_test_pred[:, 1])
mse_y2 = mean_squared_error(Y_test['Cooling Load'], Y_test_pred[:, 1])
r2_y2 = r2_score(Y_test['Cooling Load'], Y_test_pred[:, 1])

# Display evaluation metrics for Y1 and Y2
st.subheader("Model Evaluation Metrics")
st.write(f"MAE for Heating Load: {mae_y1:.2f}")
st.write(f"MSE for Heating Load: {mse_y1:.2f}")
st.write(f"R² for Heating Load: {r2_y1:.2f}")

st.write(f"MAE for Cooling Load: {mae_y2:.2f}")
st.write(f"MSE for Cooling Load: {mse_y2:.2f}")
st.write(f"R² for Cooling Load: {r2_y2:.2f}")

st.write("""
These metrics give us a sense of how well our model is performing. 
We can further improve the model by tuning hyperparameters, adding or removing features, or trying different regression algorithms.
""")

# Step 4: Residual Analysis and Error Distribution
st.header("4. Residual Analysis and Error Distribution")
st.write("""
Residual analysis helps us understand where the model might be making errors. 
By plotting the residuals (the difference between actual and predicted values), we can assess if there are any patterns that our model missed.
""")

# Calculate residuals
residuals_y1 = Y_test['Heating Load'] - Y_test_pred[:, 0]
residuals_y2 = Y_test['Cooling Load'] - Y_test_pred[:, 1]

# Plot Residuals vs Predicted Values for Y1 and Y2
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

sns.scatterplot(x=Y_test_pred[:, 0], y=residuals_y1, ax=ax[0])
ax[0].axhline(y=0, color='r', linestyle='--')
ax[0].set_xlabel('Predicted Heating Load')
ax[0].set_ylabel('Residuals')
ax[0].set_title('Residuals vs. Predicted Values for Heating Load')

sns.scatterplot(x=Y_test_pred[:, 1], y=residuals_y2, ax=ax[1])
ax[1].axhline(y=0, color='r', linestyle='--')
ax[1].set_xlabel('Predicted Cooling Load')
ax[1].set_ylabel('Residuals')
ax[1].set_title('Residuals vs. Predicted Values for Cooling Load')

st.pyplot(fig)

st.write("""
If the residuals are randomly distributed around zero, it suggests that our model has a good fit. 
However, any patterns in the residuals could indicate areas where the model is underperforming.
""")

# Plot Error Distribution for Y1 and Y2
st.subheader("Error Distribution")
st.write("""
Visualizing the distribution of errors (residuals) gives us an idea of how the model's predictions deviate from the actual values. 
A normal distribution of errors suggests a well-fitted model.
""")

fig, ax = plt.subplots(2, 1, figsize=(10, 12))

sns.histplot(residuals_y1, kde=True, ax=ax[0])
ax[0].set_title('Distribution of Prediction Errors for Heating Load')

sns.histplot(residuals_y2, kde=True, ax=ax[1])
ax[1].set_title('Distribution of Prediction Errors for Cooling Load')

st.pyplot(fig)

# Plot Actual vs Predicted Values for Y1 and Y2
st.subheader("Actual vs Predicted Values")
st.write("""
Finally, let's compare the actual values with the predicted values to visually assess the model's accuracy. 
The closer the points are to the diagonal line, the better the model's predictions.
""")

fig, ax = plt.subplots(2, 1, figsize=(10, 12))

sns.scatterplot(x=Y_test['Heating Load'], y=Y_test_pred[:, 0], ax=ax[0])
ax[0].plot([Y_test['Heating Load'].min(), Y_test['Heating Load'].max()], [Y_test['Heating Load'].min(), Y_test['Heating Load'].max()], 'r--')
ax[0].set_xlabel('Actual Heating Load')
ax[0].set_ylabel('Predicted Heating Load')
ax[0].set_title('Actual vs. Predicted Values for Heating Load')

sns.scatterplot(x=Y_test['Cooling Load'], y=Y_test_pred[:, 1], ax=ax[1])
ax[1].plot([Y_test['Cooling Load'].min(), Y_test['Cooling Load'].max()], [Y_test['Cooling Load'].min(), Y_test['Cooling Load'].max()], 'r--')
ax[1].set_xlabel('Actual Cooling Load')
ax[1].set_ylabel('Predicted Cooling Load')
ax[1].set_title('Actual vs. Predicted Values for Cooling Load')

st.pyplot(fig)

st.write("""
Thank you for using the Energy Consumption Forecasting App! 
This app provides insights into the energy efficiency of buildings and helps in predicting energy consumption accurately.
""")
