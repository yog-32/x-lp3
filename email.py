# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# %%
df = pd.read_csv('emails.csv')
print(df.head())

# %%
X = df.iloc[:, 1:3001]
y = df.iloc[:, 3001]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# %%
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# %%
# Making predictions with the SVM model
y_pred_svm = svm.predict(X_test)

# %%
# Making predictions with the KNN model
y_pred_knn = knn.predict(X_test)

print("KNN Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred_knn))

# %%
print("SVM Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred_svm))

# %%
print("KNN Classification Report:")
print(metrics.classification_report(y_test, y_pred_knn))
print("SVM Classification Report:")
print(metrics.classification_report(y_test, y_pred_svm))

# %%
print("SVM Accuracy:", metrics.accuracy_score(y_test, y_pred_svm))
print("KNN Accuracy:", metrics.accuracy_score(y_test, y_pred_knn))


