import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 1. Our Dataset (Notice the None/NaN for a missing humidity value)
data = {
    'Temp': [25, 30, 15, 20, 22, 28, 12, 18],
    'Humidity': [80, 85, 90, 60, None, 75, 95, 70],
    'Wind': [5, 10, 20, 5, 8, 12, 25, 15],
    'Is_Sunny': [1, 1, 0, 1, 1, 1, 0, 0]
}
df = pd.DataFrame(data)

# 2. Handle the Missing Value (Imputation)
# Let's fill the missing humidity with the average humidity
df['Humidity'] = df['Humidity'].fillna(df['Humidity'].mean())

# 3. Features and Target
X = df[['Temp', 'Humidity', 'Wind']]
y = df['Is_Sunny']

# 4. Split the data (The 'Final Exam' setup)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train a Decision Tree
clf = DecisionTreeClassifier(max_depth=3)  # max_depth helps prevent overfitting!
clf.fit(X_train, y_train)

print(f"Model accuracy on test set: {clf.score(X_test, y_test) * 100}%")
