import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the basketball dataset
data = pd.read_csv('basketball_dataset.csv')

# Print the column names up to Player J or use print(data.head()) to print up to Player E
print(data)

# Separate features and target variable
X = data.drop(['Player', 'ShotResult'], axis=1)
y = data['ShotResult']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



