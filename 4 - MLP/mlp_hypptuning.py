## MLP - Hyperparameters tuning

# Each partition: 1m 30s

############################################################
## USER PARAMETERS: model hyperparameters tuning grid
WIDTH_GRID = [2, 32]#[2, 8, 32, 128, 256]
LENGTH_GRID = [4,32]#[2, 4, 8, 32]
LR_GRID = [1e-4,1e-1]#[1e-4, 1e-3, 1e-2, 1e-1]
MOMENTUM_GRID = [.5,1.]#[.5, .8, .9, 1.]
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

# Hyperparameters tuning dataset
DATASET = 'swarm_lda.csv'
SMALL = False

# Load dataset and train-val partition
small = '' if not SMALL else 'S'
path_DS = '../data/datasets/csv/'
path_indices = '../data/partitions/csv/'
df_np = pd.read_csv(path_DS + DATASET).to_numpy()

# Device: not necessary, it can run well in CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model
class MLP(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers):
    super(MLP, self).__init__()
    self.length = num_layers
    fcs = []
    fcs.append(nn.Linear(input_size, hidden_size))
    for _ in range(self.length - 2):
      fcs.append(nn.Linear(hidden_size, hidden_size))
    fcs.append(nn.Linear(hidden_size, output_size))
    self.fcs = nn.ModuleList(fcs)
    self.activation = nn.ReLU(inplace=True)
  
  def forward(self, x):
    x = x.float()
    for i, fc in enumerate(self.fcs):
      x = fc(x)
      if i != self.length - 1:
        x = self.activation(x)
    return x

# TRAINING
AUC_hyperp = []

for PARTITION in range(1, 11):
  AUC_hyperp_partition = []

  # Training-validation partition (no test in the hyperparameters tuning)
  train_indices = pd.read_csv(path_indices + 'iTrain'+small+str(PARTITION)+'.csv').squeeze()
  val_indices = pd.read_csv(path_indices + 'iVal'+small+str(PARTITION)+'.csv').squeeze()

  X_training, X_validation = df_np[train_indices, :-1], df_np[val_indices, :-1]
  y_training, y_validation = df_np[train_indices, -1], df_np[val_indices, -1]

  # Convert to tensor
  X_trainingCPU, y_trainingCPU = torch.tensor(X_training, requires_grad=True), torch.tensor(y_training, requires_grad=False)
  X_validationCPU, y_validationCPU = torch.tensor(X_validation, requires_grad=False), torch.tensor(y_validation, requires_grad=False)

  # Select to device (CUDA if possible)
  X_trainingDEV, y_trainingDEV = X_trainingCPU.to(device), y_trainingCPU.to(device)
  X_validationDEV, y_validationDEV = X_validationCPU.to(device), y_validationCPU.to(device)

  for width in WIDTH_GRID:
    # print(f'width {width}')
    for length in LENGTH_GRID:
      # print(f'length {length}')
      for lr in LR_GRID:
        # print(f'lr {lr}')
        for momentum in MOMENTUM_GRID:
          # print(f'momentum {momentum}')
          torch.manual_seed(0)
          model = MLP(X_training.shape[1], width, 2, length).to(device)
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

          # Testing on validation set (the hyperparameters tuning process shouldn't "see" the test set)
          model.eval()
          with torch.no_grad():
            y_hat_validation = model.forward(X_validationDEV)
            predictions = y_hat_validation.argmax(dim=1)
            AUC = roc_auc_score(y_validationDEV.to('cpu'), predictions.to('cpu'))
            AUC_hyperp_partition.append(
              (
                AUC,
                width,
                length,
                lr,
                momentum,
              )
            )
            
  print(f'Max AUC on partition {PARTITION}:', 
    {np.max([i[0] for i in AUC_hyperp_partition], axis=0)})

  AUC_hyperp.append(max(AUC_hyperp_partition))

AUC_hyperp = np.array(AUC_hyperp)

for AUC, width, length, lr, momentum in AUC_hyperp:
  print(AUC, width, length, lr, momentum)

modes = stats.mode(AUC_hyperp[:, 1:])[0][0]
hypps = ['Width', 'Length', 'LR', 'Momentum']
assert len(modes) == len(hypps)
print('Hyperparameters selection:')

for i in range(len(hypps)):
  print(f'\t{hypps[i]}\t{modes[i]}')





































