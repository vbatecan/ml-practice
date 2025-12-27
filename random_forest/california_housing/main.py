import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. Load Data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# # 2. Scale the Features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
reg_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
reg_model.fit(X_train, y_train)

# 5. Evaluate
score = reg_model.score(X_test, y_test)
print(f"Score: {score*100:.2f}%")
predictions = reg_model.predict(X_test)
print(f"R-squared Score: {r2_score(y_test, predictions):.4f}")