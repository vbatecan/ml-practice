import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "experience": [
        2,
        5,
        1,
        10
    ],
    "education_level": [
        "Bachelor",
        "Master",
        "Bachelor",
        "PhD"
    ],
    "department": [
        "Engineering",
        "HR",
        "Marketing",
        "Research"
    ],
    "salary": [
        40000,
        50000,
        35000,
        90000
    ]
}

df = pd.DataFrame(data)
# Separate features
features = df[["experience", "education_level", "department"]]

# Use get_dummies to convert categorical variables into dummy/indicator variables
education_dummies = pd.get_dummies(features[["education_level"]], dtype="int")
department_dummies = pd.get_dummies(features[["department"]], dtype="int")
print(features)

features = features.drop("department", axis=1)
features = features.drop("education_level", axis=1)
print(features)

x = pd.concat([features, education_dummies, department_dummies], axis=1)
print(x)

y = pd.DataFrame(data["salary"])
print(y)

model = LinearRegression()
model.fit(x, y)

data_in = pd.DataFrame([
    [4, 1, 0, 0, 1, 0, 0, 0]
], columns=x.columns)

predictions = model.predict(data_in)
print(f"Expected Salary is: ${predictions[0][0]:.2f}")
