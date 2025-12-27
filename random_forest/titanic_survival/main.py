import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# This URL contains the famous Titanic training data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.DataFrame(pd.read_csv(url))

imputer = SimpleImputer(strategy='mean')
embarked_imputer = SimpleImputer(strategy="most_frequent")

# Drop irrelevant columns and infer features
df = df.drop(columns=['PassengerId', 'Ticket', 'Cabin'])
df['Title'] = df['Name'].str.extract(r', (\w+)\.', expand=False)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

title_ages = df.groupby("Title")["Age"].mean()
df["Age"] = df.apply(lambda row:
                     title_ages[row["Title"]] if pd.isnull(row["Age"]) else row["Age"],
                     axis=1
                     )

# Impute values
df[["Age"]] = imputer.fit_transform(df[["Age"]])
df[["Embarked"]] = embarked_imputer.fit_transform(df[["Embarked"]])

# Encoding
standardizer = StandardScaler()
df[["Age", "Fare"]] = standardizer.fit_transform(df[["Age", "Fare"]])

# Get features and categorical features
title_dummies = pd.get_dummies(df[['Title']])
sex_dummies = pd.get_dummies(df[["Sex"]], dtype="int")
embarked_dummies = pd.get_dummies(df[["Embarked"]], dtype="int")

x = df.drop(columns=['Survived', 'Sex', 'Embarked', 'Title', 'Name'], axis=1)
x = pd.concat([x, title_dummies, sex_dummies, embarked_dummies], axis=1)
y = df['Survived']

# Split train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
model.fit(x_train, y_train)
print(x.columns)
print(model.feature_importances_)
print(f"Score: {model.score(x_test, y_test) * 100:.2f}%")

param_grid = {
    'n_estimators': [10, 25, 50, 75, 100, 150, 200],
    'max_depth': [3, 5, 7, 9, 15, 20, 25]
}

# Let's find the best n_estimators and max_depth
grid = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(x_train, y_train)
print("Best params", grid.best_params_)
print("Best score", grid.best_score_)
