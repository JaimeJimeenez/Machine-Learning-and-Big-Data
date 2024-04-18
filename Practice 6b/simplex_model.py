import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from data import *

class SimplexModel(nn.Module):
    def __init__(self):
        super(SimplexModel, self).__init__()
        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 6)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_simplex_model():
    print('Inicio del entrenamiento')
    X, y = generate_data()
    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.5, random_state=1)
    X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.2, random_state=1)

    X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
    X_train_norm = torch.from_numpy(X_train_norm).float()
    y_train = torch.from_numpy(y_train)

    train_ds = TensorDataset(X_train_norm, y_train)
    torch.manual_seed(1)
    batch_size = 6
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    input_size = X_train_norm.shape[1]
    hidden_size = 40
    output_size = 6

    model = SimplexModel()

    learning_rate = 0.01
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

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
            is_correct = (torch.argmax(pred, dim = 1) == y_batch).float()
            accuracy_hist[epoch] += is_correct.sum()

        loss_hist[epoch] /= len(train_dl.dataset)
        accuracy_hist[epoch] /= len(train_dl.dataset)
        if epoch % log_epochs == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch} loss {loss_hist[epoch]:.4f}') 

    display_accuracy_loss(accuracy_hist, loss_hist)
    X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
    X_test_norm = torch.from_numpy(X_test_norm).float()
    y_test = torch.from_numpy(y_test)

    pred_test = model(X_test_norm)

    correct = (torch.argmax(pred_test, dim = 1) == y_test).float()
    accuracy = correct.mean()

    print(f'Test accuracy: {accuracy:.4f}')
    
train_simplex_model()