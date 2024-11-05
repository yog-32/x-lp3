# %%
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# %%
data = pd.read_csv('bank.csv')
print(data.head())

# %%
# Step 3: Prepare features and target
# Selecting relevant columns for features and target variable
X = data.iloc[:, 3:13].values  
y = data.iloc[:, 13].values    


# %%
# Encode categorical data (e.g., Geography and Gender)
label_encoder_geo = LabelEncoder()
X[:, 1] = label_encoder_geo.fit_transform(X[:, 1])  

label_encoder_gender = LabelEncoder()
X[:, 2] = label_encoder_gender.fit_transform(X[:, 2])  

# %%
# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
# Step 5: Initialize and build the neural network model
model = Sequential()

# Input layer and first hidden layer
model.add(Dense(units=6, activation='relu', input_shape=(X_train.shape[1],)))

# Adding second hidden layer
model.add(Dense(units=6, activation='relu'))

# %%
# Output layer (binary classification, so 1 output neuron with sigmoid activation)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# %%
# Step 6: Evaluate the model
# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary (0 or 1)

# %%
# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


