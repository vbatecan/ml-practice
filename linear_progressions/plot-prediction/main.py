import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([[100], [200], [300], [400], [500]])
y = np.array([1500, 3000, 4500, 6000, 7500])

model = LinearRegression()
model.fit(x, y)

sqft = int(input("Enter sqft : "))
prediction = model.predict(
    np.array([[sqft]])
)
print(f"The predicted price is: {prediction[0]:.2f}")
