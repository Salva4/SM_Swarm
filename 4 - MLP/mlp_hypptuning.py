## MLP - Hyperparameters tuning

############################################################
## USER PARAMETERS: model hyperparameters tuning grid
WIDTH_GRID = [2, 8, 32, 128]
DEPTH_GRID = [1, 2, 4, 8, 32] 
LR_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
MOMENTUM_GRID = [0., .5, .9, 1.5, 5.] 

## USER INPUTS
DATASET = input('''- Write the name of the dataset:
  original / lda / pca / autoenc / pca_corr1 / pca_corr2 / pca_corr3\n--> ''')
assert DATASET in ['original', 'lda', 'pca', 'autoenc', 'pca_corr1', 'pca_corr2', 'pca_corr3']

SMALL = input('Small dataset? (only 10000 first samples) no / yes\n--> ') if 'pca' in DATASET else 'no'
assert SMALL in ['yes', 'no']

BALANCED = input('Balanced dataset? no / yes\n--> ') \
  if (SMALL == 'no') and ('corr' not in DATASET) else 'no'
assert BALANCED in ['yes', 'no']
############################################################

# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
from sklearn.metrics import roc_auc_score
from scipy import stats

PARTITION = 1   # Hyperparameter tuning done on Partition 1

# Load dataset and train-val-test partition
ds_file = DATASET if DATASET != 'lda' else 'lda/lda_1' if BALANCED == "no" else 'lda/balanced_lda_1' 
ds_file = ds_file if BALANCED == 'no' else 'balanced_' + ds_file
ds_file += '.csv'
small = '' if SMALL == 'no' else 'S'
balanced = '' if BALANCED == 'no' else 'B'
path_DS = '../data/datasets/csv/'
path_indices = '../data/partitions/csv/'
df_np = pd.read_csv(path_DS + ds_file).to_numpy()

# Device: not necessary, it can run well in CPU
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

# Training-validation partition (no test in the hyperparameters tuning)
train_indices = pd.read_csv(path_indices + 'iTrain'+small+balanced+str(PARTITION)+'.csv').squeeze()
val_indices = pd.read_csv(path_indices + 'iVal'+small+balanced+str(PARTITION)+'.csv').squeeze()

X_training, X_validation = df_np[train_indices, :-1], df_np[val_indices, :-1]
y_training, y_validation = df_np[train_indices, -1], df_np[val_indices, -1]

# Convert to tensor
X_trainingCPU, y_trainingCPU = torch.tensor(X_training, requires_grad=True), torch.tensor(y_training, requires_grad=False)
X_validationCPU, y_validationCPU = torch.tensor(X_validation, requires_grad=False), torch.tensor(y_validation, requires_grad=False)

# Select to device (CUDA if possible)
X_trainingDEV, y_trainingDEV = X_trainingCPU.to(device), y_trainingCPU.to(device)
X_validationDEV, y_validationDEV = X_validationCPU.to(device), y_validationCPU.to(device)

# TRAINING
AUC_hyperp = []

for width in WIDTH_GRID:
  for depth in DEPTH_GRID:
    for lr in LR_GRID:
      for momentum in MOMENTUM_GRID:
        torch.manual_seed(0)
        model = MLP(X_training.shape[1], width, 2, depth).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
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

        # "Testing" on validation set (the hyperparameters tuning process shouldn't "see" the test set)
        model.eval()
        with torch.no_grad():
          y_hat_validation = model.forward(X_validationDEV)
          predictions = y_hat_validation.argmax(dim=1)
          AUC = roc_auc_score(y_validationDEV.to('cpu'), predictions.to('cpu'))
          AUC_hyperp.append(
            (
              AUC,
              width,
              depth,
              lr,
              momentum,
            )
          )

selection = max(AUC_hyperp)

print(f'\nHyperparameters selection: (AUC: {selection[0]})')

hypps = ['Width', 'Depth', 'LR', 'Momentum']
for i in range(1, len(selection)):
  print(f'\t{hypps[i-1]}\t{selection[i]}')

































