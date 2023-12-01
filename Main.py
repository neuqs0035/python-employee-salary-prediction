import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = pd.read_csv("Salary_Data.csv")

X = data[["Experience_Years", "Profession_Code", "Education_Level_Code"]]
y = data["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


new_data_point = pd.DataFrame({
    "Experience_Years": [5],
    "Profession_Code": [2],
    "Education_Level_Code": [3]
})

predicted_salary = model.predict(new_data_point)

print("Predicted Salary:", predicted_salary[0])