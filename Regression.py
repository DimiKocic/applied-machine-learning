import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from scipy.stats import zscore
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoCV
import sys

if sys.stdout.encoding != 'UTF-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')



pd.set_option('display.max_columns', 2)

pd.set_option('display.expand_frame_repr', False)

# Load dataset
df = pd.read_csv(r'.\Regression\melb_data.csv', header=0, sep=",")

# Fill missing values for 'Car' with the mean per 'Suburb'
df['Car'] = df['Car'].fillna(df.groupby('Suburb')['Car'].transform('mean').round())

df['YearBuilt'] = df['YearBuilt'].fillna(df.groupby('Suburb')['YearBuilt'].transform('mean').round())

df.drop(df[df['YearBuilt'].isnull()].index, inplace=True)


#np.random.seed(42)

# Step 1: Calculate mean per Regionname
mean_building_area = df.groupby('Regionname')['BuildingArea'].mean()

# Step 2: Calculate bounds per Regionname
lower_bounds = mean_building_area * 0.5
upper_bounds = mean_building_area * 1.5

def fill_missing_area(row):
    if pd.isna(row['BuildingArea']):
        region = row['Regionname']
        lb = lower_bounds.get(region, 0) 
        ub = upper_bounds.get(region, 0)
        return np.random.uniform(lb, ub)
    else:
        return row['BuildingArea']

df['BuildingArea'] = df.apply(fill_missing_area, axis=1)



df.drop(columns =['Suburb' , 'Address', 'SellerG','Postcode','CouncilArea','Lattitude','Longtitude','Date'], inplace=True)

df.drop(df[df['Method'] == 'SA'].index, inplace=True)

missing_values = pd.DataFrame({'percent_missing': df.isnull().sum() * 100 / len(df)})
#print(missing_values)

categorical_col = [col for col in df.columns if df[col].dtype == 'object']
#print("Categorical columns:", categorical_col)


threshold = 8  # If a category appears in less than % of the data, we group it as "Other"

# Print category percentages for categorical columns
for col in categorical_col:
    category_percentages = df[col].value_counts(normalize=True) * 100
    #print(f"Category percentages for {col}:\n{category_percentages}\n")
    rare_categories = category_percentages[category_percentages < threshold].index.tolist()  # Get rare categories
    # Replace rare categories with "Other"
    df[col] = df[col].replace(rare_categories, "Other")

# for col in categorical_col:
#     print(f"Updated category counts for {col}:")
#     print(df[col].value_counts())
#     print()  # For better readability


data_encoded = pd.get_dummies(df, columns=categorical_col, drop_first=False)

# Move 'Price' column to the end
data_encoded = data_encoded[[col for col in data_encoded.columns if col != 'Price'] + ['Price']]

# Move 'Price' column to the end
data_encoded = data_encoded[[col for col in data_encoded.columns if col != 'Price'] + ['Price']]

data_encoded['Z_' + 'Price'] = zscore(data_encoded['Price'])



outliers = np.abs(data_encoded['Z_' + 'Price']) > 3
data_encoded = data_encoded[~outliers]

data_encoded = data_encoded.drop('Z_' + 'Price', axis=1)

# Separate features and target
X = data_encoded.iloc[:, :-1]
Y = data_encoded.iloc[:, -1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Linear
regr_linear = linear_model.LinearRegression()
regr_linear.fit(X_train, y_train)

# Make predictions
y_pred = regr_linear.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#Linear Results
print("\n📊 Linear Regression Evaluation")
print("-" * 40)
print(f"📌 Intercept: {regr_linear.intercept_:.2f}")
print(f"📌 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📌 Mean Squared Error (MSE): {mse:.2f}")
print(f"📌 Explained Variance (t Score): {r2:.2f}")
print("\n🧮 Coefficients:")
for feature, coef in zip(X.columns, regr_linear.coef_):
    print(f"  {feature:20}: {coef:.4f}")

# LassoCV with 5-fold cross-validation
regr_lasso = LassoCV(cv=5, max_iter=10000, n_alphas=100, random_state=42)
regr_lasso.fit(X_train, y_train)

# Predictions
y_pred_lasso = regr_lasso.predict(X_test)

# Evaluation
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\n🧩 LassoCV Regression Evaluation")
print("-" * 40)
print(f"📌 Intercept: {regr_lasso.intercept_:.2f}")
print(f"📌 Optimal Alpha: {regr_lasso.alpha_:.2f}")
print(f"📌 Mean Absolute Error (MAE): {mae_lasso:.2f}")
print(f"📌 Mean Squared Error (MSE): {mse_lasso:.2f}")
print(f"📌 Explained Variance (R² Score): {r2_lasso:.2f}")
print("\n🧮 Coefficients:")
for feature, coef in zip(X.columns, regr_lasso.coef_):
    print(f"  {feature:20}: {coef:.4f}")
    
    
    
# Polynomial   
polyDegree = 2
poly = PolynomialFeatures(degree=polyDegree)

# Transform both the training and test sets
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train Polynomial Regression Model
regr_poly = linear_model.LinearRegression()
regr_poly.fit(X_train_poly, y_train)

# Make predictions using the testing set
y_pred_poly = regr_poly.predict(X_test_poly)

# Evaluation
mse_poly = mean_squared_error(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# Print results for Polynomial Regression
print("\n📊 Polynomial Regression (Degree = 2) Evaluation")
print("-" * 40)
print(f"📌 Intercept: {regr_poly.intercept_:.2f}")
print(f"📌 Mean Absolute Error (MAE): {mae_poly:.2f}")
print(f"📌 Mean Squared Error (MSE): {mse_poly:.2f}")
print(f"📌 Explained Variance (R² Score): {r2_poly:.2f}")
# Apply cross-validation for stability
scores = cross_val_score(regr_poly, X_train_poly, y_train, cv=5, scoring='r2')
print(f"Cross-validated R² scores: {scores}")
print(f"Mean R² score: {scores.mean():.2f}")



# PLots
# Create a figure with subplots for each model
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Linear Regression Plot (Actual vs Predicted)
axes[0].scatter(y_test, y_pred, color='blue', alpha=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # Ideal line
axes[0].set_title('Linear Regression: Actual vs Predicted')
axes[0].set_xlabel('Actual Values')
axes[0].set_ylabel('Predicted Values')

# Lasso Regression Plot (Actual vs Predicted)
axes[1].scatter(y_test, y_pred_lasso, color='green', alpha=0.5)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # Ideal line
axes[1].set_title('Lasso Regression: Actual vs Predicted')
axes[1].set_xlabel('Actual Values')
axes[1].set_ylabel('Predicted Values')

# Polynomial Regression Plot (Actual vs Predicted)
axes[2].scatter(y_test, y_pred_poly, color='purple', alpha=0.5)
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # Ideal line
axes[2].set_title('Polynomial Regression: Actual vs Predicted')
axes[2].set_xlabel('Actual Values')
axes[2].set_ylabel('Predicted Values')

# Adjust layout for better readability
plt.tight_layout()

# Show the plots
plt.savefig("All_Models.png")



# Linear Regression Plot (separate)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.savefig("Linear.png")


# Lasso Regression Plot (separate)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_lasso, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title("Lasso Regression: Actual vs Predicted")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.savefig("Lasso.png")


# Polynomial Regression Plot (separate)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_poly, color='orange')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title("Polynomial Regression: Actual vs Predicted")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.savefig("Polynomial.png")
