from math import log1p

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime
import math


def datefloat_datetime(df: float):
    year = int(df)
    fraction = df - year
    days_in_year = (datetime.date(year + 1, 1, 1) - datetime.date(year, 1, 1)).days
    day_of_year = math.floor(fraction * days_in_year)
    date = datetime.date(year, 1, 1) + datetime.timedelta(days=day_of_year)
    return date


frame = pd.read_excel("data.xlsx")

# Preprocessing
# frame = frame.drop(columns=["latitude", "longitude"])
frame["year"] = frame["transaction_date"].apply(lambda row: datefloat_datetime(row).year)
frame["month"] = frame["transaction_date"].apply(lambda row: datefloat_datetime(row).month)
frame["convenience_store_log"] = frame["convenience_store"].apply(lambda row: log1p(row))
frame["distance_mrt_log"] = frame["distance_mrt"].apply(lambda row: log1p(row))
frame["distance_mrt_x_longitude"]  = frame[["longitude", "distance_mrt"]].apply(lambda row: row["distance_mrt"] + row["longitude"], axis=1)
# frame["distance_mrt_x_latitude"] = frame[["latitude", "distance_mrt"]].apply(lambda row: row["distance_mrt"] + row["latitude"], axis=1)

# create age group
bins = [0, 5, 15, 30, 50]
labels = ["new", "semi_new", "old", "very_old"]
frame["age_group"] = pd.cut(frame["age"], bins=bins, labels=labels, right=True)

age_group_dummies = pd.get_dummies(frame[["age_group"]], dtype="int")
frame = frame.drop("age_group", axis=1)
frame = pd.concat([age_group_dummies, frame], axis=1)

# hindi tayo gagamit ng scaler since random forest naman gagamitin natin and random forest doesn't care about who's highest.
x = frame.drop(["price", "transaction_date"], axis=1)
y = frame["price"]

model = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(f"Score: {score * 100:.2f}%")
