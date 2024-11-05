# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


# %%
# Load the dataset
data = pd.read_csv('diabetes.csv')
print(data.head())

# %%
# Define X (features) and y (target)
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']               # Target


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# %%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %%
# Initialize KNN with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)


# %%
y_pred = knn.predict(X_test)


# %%
# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Error rate (1 - accuracy)
error_rate = 1 - accuracy

# Precision
precision = precision_score(y_test, y_pred)

# Recall
recall = recall_score(y_test, y_pred)

# %%
# Print results
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)


