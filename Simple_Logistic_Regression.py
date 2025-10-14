# Simple Logistic Regression
# Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Example data (X = features, y = labels)
# Let's say we’re predicting whether a student passes (1) or fails (0)
X = np.array([
    [35, 2],  # hours studied, previous fails
    [40, 1],
    [10, 5],
    [25, 3],
    [50, 0],
    [5, 4],
    [45, 1]
])
y = np.array([1, 1, 0, 0, 1, 0, 1])  # labels: pass or fail

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict on new data
new_student = np.array([[30, 2]])  # 30 hours studied, 2 previous fails
print("Predicted outcome (1=Pass, 0=Fail):", model.predict(new_student)[0])
