# import requirements
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Create dataset
data = {
    'Budget': [10, 50, 20, 100, 30],
    'Is_Action': [0, 1, 0, 1, 0],
    'Revenue': [25, 120, 45, 250, 70]
}

df = pd.DataFrame(data)

x = df[["Budget", "Is_Action"]]
y = df["Revenue"]

# Now we need to scale the budget
scaler = StandardScaler()
x_scaled = x.copy()
x_scaled[["Budget"]] = scaler.fit_transform(x[["Budget"]])
print(x_scaled)

model = LinearRegression()
model.fit(x_scaled, y)

# Data
data_in = pd.DataFrame([[10, 0]], columns=["Budget", "Is_Action"])
data_in[['Budget']] = scaler.transform(data_in[["Budget"]])
prediction = model.predict(data_in)

print(f"The predicted revenue is: {prediction[0]:.2f}")