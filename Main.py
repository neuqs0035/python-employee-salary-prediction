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

experience_year = float(input("Enter Year Of Experience You Have : "))
education_level_code = int(input("1] Bachelors Degree\n2] Masters Degree\n3] Phd\nEnter From Above Choice (number) _ : "))

profession_code = int(input("1] Software Developer\n2] Data Analyst\n3] Registered Nurse\n4] Teacher\n5] Accountant\n6] Marketing Manager\n7] Mechanical Engineer\n8] Medical Doctor\n9] Graphic Designer\n10] Electrician\nEnter From Above Choice (number) _ : "))

new_data_point = pd.DataFrame({
    "Experience_Years": [experience_year],
    "Profession_Code": [profession_code],
    "Education_Level_Code": [education_level_code]
})

predicted_salary = model.predict(new_data_point)

print("Predicted Salary:", predicted_salary[0])