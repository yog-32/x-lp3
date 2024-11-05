# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# %%
data = pd.read_csv("uber.csv", sep=',', on_bad_lines='skip')

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
data = pd.read_csv("data/uber.csv", sep=',', on_bad_lines='skip')
# %%
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], errors='coerce')

# %%
missing_values = data.isnull().sum()
print("Missing values in the dataset:")
print(missing_values)

# %%
data.dropna(inplace=True)
missing_values = data.isnull().sum()
print("Missing values after handling:")
print(missing_values)

# %%
data['fare_amount'] = data['fare_amount'].astype(str)
sns.boxplot(x=data['fare_amount'])
plt.show()

# %%
# Calculate the IQR for the 'fare_amount' column
# Convert 'fare_amount' back to numeric type
data['fare_amount'] = pd.to_numeric(data['fare_amount'])
Q1 = data["fare_amount"].quantile(0.25)
Q3 = data["fare_amount"].quantile(0.75)
IQR = Q3 - Q1

# %%
# Define a threshold (e.g., 1.5 times the IQR) to identify outliers
threshold = 1.5
lower_bound = Q1 - threshold * IQR
upper_bound = Q3 + threshold * IQR
# Remove outliers
data_no_outliers = data[(data["fare_amount"] >= lower_bound) & (data["fare_amount"])]
data.plot(kind="box",subplots=True, layout=(7, 2), figsize=(15, 20))

# %%
# 3. Check the correlation
# Determine the correlation between features and the target variable (fare_amount).
# Drop non-numeric columns before calculating correlation
numeric_data = data.drop(columns=['pickup_datetime', 'key'])  
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()



# %%
from math import radians, cos, sin, asin, sqrt

# Define a function to calculate distance based on latitude and longitude
def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

# Apply the distance calculation to create a new column
data['distance_km'] = data.apply(lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'],
                                                      row['dropoff_longitude'], row['dropoff_latitude']), axis=1)

# Define X and y
X = data[['distance_km']]  # Use calculated distance as the feature
y = data['fare_amount']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)




# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Create and train the Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions with the Random Forest model
y_pred_rf = rf_model.predict(X_test)

# Calculate R2 and RMSE for Random Forest
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)

# Print the results
print("Random Forest Regression R2:", r2_rf)
print("Random Forest Regression RMSE:", rmse_rf)



