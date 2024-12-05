import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Feature vectors (each row is a feature vector)
X = np.array([
    [26, 32, 35, 31, 31, 32],   
    [193, 193, 231, 238, 216, 0],  
    [183, 183, 183, 150, 156, 161]  
])

# Labels for the feature vectors
y = np.array([0, 1, 2])  # Example labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define and train the MLP model
model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Predicting new data
new_data = np.array([[26, 32, 35, 31, 31, 32]])  # Example new feature vector
prediction = model.predict(new_data)
print(f"Prediction for new data: {prediction[0]}")