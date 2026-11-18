# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
In educational institutions, understanding the direct impact of study habits on academic performance is crucial. The goal of this project is to build a predictive system that takes a single continuous feature—the number of hours a student spends studying—and outputs a continuous numerical prediction of their expected final exam score. Because the target variable (marks) is a continuous numerical value rather than a discrete category, this challenge is classified as a regression problem

## Neural Network Model
Include the neural network model diagram.
<img width="1013" height="618" alt="image" src="https://github.com/user-attachments/assets/a02cfad7-e35b-4e72-abf5-308f69f791e4" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Bharath K

### Register Number: 212224230036

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("student_study_hours_regression.csv")

print(df)
X = df[["Study_Hours"]].values
y = df[["Exam_Marks"]].values

print("Input:")
print(X)

print("Output:")
print(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)
print(X_test)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(X_train.shape)
print(y_train.shape)
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)
torch.manual_seed(33)

model = NeuralNet()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(model)
epochs = 1000
loss_list = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
model.eval()

with torch.no_grad():
    predictions = model(X_test).numpy()

print("\nModel Evaluation")
print("MSE:", mean_squared_error(y_test.numpy(), predictions))
print("RMSE:", mean_squared_error(y_test.numpy(), predictions) ** 0.5)
print("MAE:", mean_absolute_error(y_test.numpy(), predictions))
print("R2 Score:", r2_score(y_test.numpy(), predictions))
plt.figure(figsize=(8, 4))
plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Student Marks Prediction - Training Loss")
plt.show()
new_student = [[9]]

new_student_scaled = scaler.transform(new_student)

new_student_tensor = torch.tensor(
    new_student_scaled,
    dtype=torch.float32
)

with torch.no_grad():
    predicted_marks = model(new_student_tensor).item()

print(f"Predicted Exam Marks for 9 Study Hours: {predicted_marks:.2f}")
    

```

### Dataset Information

<img width="720" height="491" alt="image" src="https://github.com/user-attachments/assets/23496260-3ceb-4de8-80de-2773c17bd048" />

### OUTPUT

### Training Loss Vs Iteration Plot

<img width="532" height="228" alt="image" src="https://github.com/user-attachments/assets/47fce228-5f92-41f3-85e0-cf4887e368a5" />

### Lost List

<img width="662" height="576" alt="image" src="https://github.com/user-attachments/assets/5f2a69af-27b2-404b-98d8-5a66c758e089" />

### Model evaluation
<img width="867" height="426" alt="image" src="https://github.com/user-attachments/assets/62cca395-2616-417a-9d4c-0c4ceac20db6" />

### Plot Diagram
<img width="853" height="496" alt="image" src="https://github.com/user-attachments/assets/08739f91-9282-46eb-811e-204fce9018a8" />


### New Sample Data Prediction

<img width="682" height="366" alt="image" src="https://github.com/user-attachments/assets/7f61f2d6-56d7-4982-84ea-069c82d970d9" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
