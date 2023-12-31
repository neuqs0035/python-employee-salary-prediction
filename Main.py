import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = pd.read_csv("Salary_Data.csv")

print("\nDataset Loaded ........")

X = data[["Experience_Years", "Profession_Code", "Education_Level_Code"]]
y = data["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

print("Training Model ........")

model.fit(X_train, y_train)

print("Model Trained Successfully ........")
print("Testing Model Performance ........")
y_pred = model.predict(X_test)
print("Model Testing Completed ........")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

print("\n- - - - - - - - New Prediction - - - - - - - -")

experience_year = float(input("\nEnter Year Of Experience You Have : "))

print("\nChoose Your Education Level : ")
education_level_code = int(input("\n1] Bachelors Degree\n2] Masters Degree\n3] Phd\n\nEnter From Above Choice (number) _ : "))

print("\nChoose You Job Titile : ")
profession_code = int(input("\n1] Software Developer\n2] Data Analyst\n3] Registered Nurse\n4] Teacher\n5] Accountant\n6] Marketing Manager\n7] Mechanical Engineer\n8] Medical Doctor\n9] Graphic Designer\n10] Electrician\n\nEnter From Above Choice (number) _ : "))

new_data_point = pd.DataFrame({
    "Experience_Years": [experience_year],
    "Profession_Code": [profession_code],
    "Education_Level_Code": [education_level_code]
})

predicted_salary = model.predict(new_data_point)

print("\nPredicted Salary:", predicted_salary[0])