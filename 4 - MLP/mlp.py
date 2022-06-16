############################################################
## USER PARAMETERS: dataset and reduced/not-reduced
# Select the desired dataset (DATASET) and whether the number of samples must be reduced to the first 10000 (SMALL).
DATASET = 'swarm_lda.csv'
SMALL = False
############################################################



import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import time
from sklearn.metrics import roc_auc_score


# Dataset and reduction (or not) of sample size
small = '' if not SMALL else 'S'
path_DS = '../data/datasets/csv/'
path_indices = '../data/partitions/csv/'
df_np = pd.read_csv(path_DS + DATASET).to_numpy()

# Device: not necessary, it can run well in CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

AUC_history, runningTime_history = [], []

for PARTITION in range(1, 11):
  # Training-validation-testing partition
  train_indices = pd.read_csv(path_indices + 'iTrain'+small+str(PARTITION)+'.csv').squeeze()
  val_indices = pd.read_csv(path_indices + 'iVal'+small+str(PARTITION)+'.csv').squeeze()
  test_indices = pd.read_csv(path_indices + 'iTest'+small+str(PARTITION)+'.csv').squeeze()

  X_training, X_validation, X_testing = df_np[train_indices, :-1], df_np[val_indices, :-1], df_np[test_indices, :-1]
  y_training, y_validation, y_testing = df_np[train_indices, -1], df_np[val_indices, -1], df_np[test_indices, -1]

  # Convert to tensor
  X_trainingCPU, y_trainingCPU = torch.tensor(X_training, requires_grad=True), torch.tensor(y_training, requires_grad=False)
  X_validationCPU, y_validationCPU = torch.tensor(X_validation, requires_grad=False), torch.tensor(y_validation, requires_grad=False)
  X_testingCPU, y_testingCPU = torch.tensor(X_testing, requires_grad=False), torch.tensor(y_testing, requires_grad=False)

  # Select to device (CUDA if possible)
  X_trainingDEV, y_trainingDEV = X_trainingCPU.to(device), y_trainingCPU.to(device)
  X_validationDEV, y_validationDEV = X_validationCPU.to(device), y_validationCPU.to(device)
  X_testingDEV, y_testingDEV = X_testingCPU.to(device), y_testingCPU.to(device)

  t0=time.time()

  class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
      super(MLP, self).__init__()
      self.fc1 = nn.Linear(input_size, hidden_size)
      self.fc2 = nn.Linear(hidden_size, hidden_size)
      self.fc3 = nn.Linear(hidden_size, hidden_size)
      self.fc4 = nn.Linear(hidden_size, output_size)
      self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
      x = x.float()
      inp_h1 = self.fc1(x)
      h1 = self.activation(inp_h1)
      inp_h2 = self.fc2(h1)
      h2 = self.activation(inp_h2)
      inp_h3 = self.fc3(h2)
      h3 = self.activation(inp_h3)
      out = self.fc4(h3)
      return out

  torch.manual_seed(0)
  model = MLP(X_training.shape[1], 5, 2)
  optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
  criterion = nn.CrossEntropyLoss()

  nITERATIONS = 1000    #40000
  LR = .01    # .0001
  history_loss_training = []
  history_loss_validation = []
  history_weights = []

  maxACCVAL = -1
  patience = 20   # lda, pca10, pca100
  #patience = 1000    # pca500
  curr_waiting = 0

  for k in range(nITERATIONS):
    model.train()
    #if k > 0:
    #  model.weight.grad.data.zero_()

    # Predictions: y_hat
    y_hat_training = model.forward(X_trainingDEV)

    loss_training = criterion(y_hat_training, y_trainingDEV.long())

    optimizer.zero_grad()
    loss_training.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
      y_hat_validation = model.forward(X_validationDEV)
      predictions = y_hat_validation.argmax(dim=1)
      accuracy_val = (predictions == y_validationDEV).sum()/predictions.shape[0]

    if accuracy_val.item() > maxACCVAL:
      maxACCVAL = accuracy_val
      curr_waiting = -1
    curr_waiting += 1

    if curr_waiting > patience:
      break

  running_time = time.time() - t0

  # Testing
  model.eval()
  with torch.no_grad():
    y_hat_testing = model.forward(X_testingDEV)
    predictions = y_hat_testing.argmax(dim=1)
    accuracy_test = (predictions == y_testingDEV).sum()/predictions.shape[0]
    AUC = roc_auc_score(y_testingDEV, predictions)
    print(f'AUC on the test set: {AUC : .16f} \t Running time: {running_time : .16f}')
    AUC_history.append(AUC)
    runningTime_history.append(running_time)

print(f'Mean AUC: {np.mean(AUC_history)}\tMean running time: {np.mean(running_time)}')