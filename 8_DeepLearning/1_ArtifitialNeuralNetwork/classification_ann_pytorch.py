## Predicting if the customer will leave the bank or not based on the data provided
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

##### Data Preprocessing
# Importing dataset
dataset = pd.read_csv(
    '/Users/adityaagrawal/PycharmProjects/PythonProject/8_DeepLearning/1_ArtifitialNeuralNetwork/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding Categorical Data --> Gender
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Encode categorical Data --> Country
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split into test and training set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)


# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


##### Building Artificial Neural Network
class ChurnPredictionNN(nn.Module):
    def __init__(self, input_size):
        super(ChurnPredictionNN, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, 6)  # First hidden layer
        self.fc2 = nn.Linear(6, 6)  # Second hidden layer
        self.fc3 = nn.Linear(6, 1)  # Output layer
        self.relu = nn.ReLU()  # ReLU activation
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for output

    def forward(self, x):
        # Forward pass
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Initialize the model
input_size = X_train.shape[1]
ann = ChurnPredictionNN(input_size)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(ann.parameters())

##### Training Artificial Neural Network
# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ann.to(device)

# Training loop
epochs = 100
ann.train()  # Set model to training mode

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = ann(batch_X)

        # Calculate loss
        loss = criterion(outputs, batch_y)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        epoch_loss += loss.item()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

##### Predict test set result
ann.eval()  # Set model to evaluation mode
X_test_tensor = X_test_tensor.to(device)

with torch.no_grad():
    y_pred_prob = ann(X_test_tensor)
    y_pred = (y_pred_prob > 0.5).float()

# Convert back to numpy for sklearn metrics
y_pred_np = y_pred.cpu().numpy().flatten()
y_test_np = y_test

# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test_np, y_pred_np)
acs = accuracy_score(y_test_np, y_pred_np)
print("Confusion Matrix:")
print(cm)
print(f"Accuracy Score: {acs:.4f}")

# Optional: Print model summary
print(f"\nModel Architecture:")
print(ann)