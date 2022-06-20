# MLP

############################################################
## USER INPUTS: dataset and reduced/not-reduced
# Select the desired dataset (DATASET) and whether the number of samples must be reduced to the first 10000 (SMALL).
DATASET = input('''- Write the name of the dataset:
  original / lda / pca / autoenc / pca_corr1 / pca_corr2 / pca_corr3\n--> ''')
assert DATASET in ['original', 'lda', 'pca', 'autoenc', 'pca_corr1', 'pca_corr2', 'pca_corr3']

SMALL = input('Small dataset? (only 10000 first samples) no / yes\n--> ') if 'pca' in DATASET else 'no'
assert SMALL in ['yes', 'no']

BALANCED = input('Balanced dataset? no / yes\n--> ') \
  if (SMALL == 'no') and ('corr' not in DATASET) else 'no'
assert BALANCED in ['yes', 'no']

# Model hyperparameters
WIDTH = int(input('NN Width: '))
DEPTH = int(input('NN Depth: '))
LR = float(input('LR: '))
MOMENTUM = float(input('SGD momentum: '))
############################################################


# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import time
from sklearn.metrics import roc_auc_score

# Load dataset and train-val-test partition
ds_file = DATASET if DATASET != 'lda' else 'lda/lda_' if BALANCED == "no" else 'lda/balanced_lda_'  # partition specified later
ds_file = ds_file if BALANCED == 'no' else 'balanced_' + ds_file
ds_file += '.csv'
small = '' if SMALL == 'no' else 'S'
balanced = '' if BALANCED == 'no' else 'B'
path_DS = '../data/datasets/csv/'
path_indices = '../data/partitions/csv/'
if DATASET != 'lda':
  df_np = pd.read_csv(path_DS + ds_file).to_numpy()

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model
class MLP(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, depth):
    super(MLP, self).__init__()
    self.depth = depth
    fcs = []
    fcs.append(nn.Linear(input_size, hidden_size))
    for _ in range(self.depth - 1):   # depth: #layers excluding input layer
      fcs.append(nn.Linear(hidden_size, hidden_size))
    fcs.append(nn.Linear(hidden_size, output_size))
    self.fcs = nn.ModuleList(fcs)
    self.activation = nn.ReLU(inplace=True)
  
  def forward(self, x):
    x = x.float()
    for i, fc in enumerate(self.fcs):
      x = fc(x)
      if i != self.depth:
        x = self.activation(x)
    return x

# TRAINING
AUC_history, runningTime_history = [], []

for partition in range(1, 11):
  if DATASET == 'lda':
    df_np = pd.read_csv(path_DS + ds_file[:-4] + str(partition) + ds_file[-4:]).to_numpy()

  # Training-validation-testing partition
  train_indices = pd.read_csv(path_indices + 'iTrain'+small+balanced+str(partition)+'.csv').squeeze()
  val_indices = pd.read_csv(path_indices + 'iVal'+small+balanced+str(partition)+'.csv').squeeze()
  test_indices = pd.read_csv(path_indices + 'iTest'+small+balanced+str(partition)+'.csv').squeeze()

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

  torch.manual_seed(0)
  model = MLP(X_training.shape[1], WIDTH, 2, DEPTH).to(device)
  optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
  criterion = nn.CrossEntropyLoss()

  nITERATIONS = 1000
  maxACCVAL = -1
  patience = 20
  curr_waiting = 0

  for k in range(nITERATIONS):
    model.train()

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
    #accuracy_test = (predictions == y_testingDEV).sum()/predictions.shape[0]
    AUC = roc_auc_score(y_testingDEV.to('cpu'), predictions.to('cpu'))
    print(f'AUC on the test set, partition {partition}: {AUC : .16f} \t Running time: {running_time : .16f}')
    AUC_history.append(AUC)
    runningTime_history.append(running_time)

print(f'Mean AUC: {np.mean(AUC_history)}\tMean running time: {np.mean(running_time)}')