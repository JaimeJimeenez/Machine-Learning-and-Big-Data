import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from data import *

class RegularizedComplexModel(nn.Module):

    def __init__(self):
        super(RegularizedComplexModel, self).__init__()
        self.fc1 = nn.Linear(2, 120)
        self.fc2 = nn.Linear(120, 40)
        self.fc3 = nn.Linear(40, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_regularized_complex_model():
    X, y = generate_data()
    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.5, random_state=1)
    X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.2, random_state=1)

    X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
    X_train_norm = torch.from_numpy(X_train_norm).float()
    y_train = torch.from_numpy(y_train)

    train_ds = TensorDataset(X_train_norm, y_train)
    torch.manual_seed(1)
    batch_size = 120
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    input_size = X_train_norm.shape[1]
    hidden_size = 40
    output_size = 6

    model = RegularizedComplexModel()

    learning_rate = 0.001
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.1)

    num_epochs = 1000
    log_epochs = num_epochs / 10
    loss_hist = [0] * num_epochs
    accuracy_hist = [0] * num_epochs

    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch.long())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_hist[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist[epoch] += is_correct.sum()
        
        loss_hist[epoch] /= len(train_dl.dataset)
        accuracy_hist[epoch] /= len(train_dl.dataset)
        if epoch % log_epochs == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch} loss {loss_hist[epoch]}')

    display_accuracy_loss(accuracy_hist, loss_hist)

    X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
    X_test_norm = torch.from_numpy(X_test_norm).float()
    y_test = torch.from_numpy(y_test)

    pred_test = model(X_test_norm)

    correct = (torch.argmax(pred_test, dim=1) == y_test).float()
    accuracy = correct.mean()

    print(f'Test accuracy: {accuracy:.4f}') 

def train_regularized_complex_model_with_reg(regularization_value):
    X, y = generate_data()
    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.5, random_state=1)
    X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.2, random_state=1)

    X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
    X_train_norm = torch.from_numpy(X_train_norm).float()
    y_train = torch.from_numpy(y_train)

    X_val_norm = (X_cv - np.mean(X_train)) / np.std(X_train)
    X_val_norm = torch.from_numpy(X_val_norm).float()
    y_val = torch.from_numpy(y_cv)

    train_ds = TensorDataset(X_train_norm, y_train)
    val_ds = TensorDataset(X_val_norm, y_val)

    batch_size = 120
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = RegularizedComplexModel()

    learning_rate = 0.001
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization_value)

    num_epochs = 1000
    log_epochs = num_epochs / 10
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch.long())
            loss.backward()
            optimizer.step()

        model.eval()
        val_accuracy = 0.0
        with torch.no_grad():
            for x_val_batch, y_val_batch in val_dl:
                val_pred = model(x_val_batch)
                val_accuracy += (torch.argmax(val_pred, dim=1) == y_val_batch).float().mean().item()
        
        val_accuracy /= len(val_dl)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state_dict = model.state_dict()

    return best_val_accuracy

regularization_values = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
best_accuracy = 0.0
best_regularization_value = None

for reg_value in regularization_values:
    val_accuracy = train_regularized_complex_model_with_reg(reg_value)
    print(f"Regularization value: {reg_value}, Validation Accuracy: {val_accuracy}")
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_regularization_value = reg_value

print(f"Best regularization value: {best_regularization_value}, Best validation accuracy: {best_accuracy}")

train_regularized_complex_model()