#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from random import sample

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score,roc_auc_score,accuracy_score,classification_report,plot_confusion_matrix,plot_roc_curve,precision_recall_curve


# In[3]:


data = pd.read_csv('Swarm_Behaviour.csv')
data.head()


# ## Balanced data

# In[4]:


from imblearn.under_sampling import RandomUnderSampler

X = data.drop("Swarm_Behaviour",axis=1)
y = data["Swarm_Behaviour"]
rus = RandomUnderSampler(random_state=42, replacement=True)
X_rus, y_rus = rus.fit_resample(X, y)


# In[13]:


balanced_data = pd.concat([X_rus,y_rus], axis=1)
balanced_data.head()


# In[11]:


X_rus


# ## Train, Test, Validation Set

# In[14]:


X = X_rus
y = y_rus

train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))     


# In[16]:


train_idx = pd.DataFrame(x_train.index)
val_idx = pd.DataFrame(x_val.index)
test_idx = pd.DataFrame(x_test.index)


# In[17]:


for _ in range(9):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    
    train_idx = pd.concat([train_idx, pd.DataFrame(x_train.index)], axis = 1)
    val_idx = pd.concat([val_idx, pd.DataFrame(x_val.index)], axis = 1)
    test_idx = pd.concat([test_idx, pd.DataFrame(x_test.index)], axis = 1)


# In[18]:


train_idx.to_csv('train_idx_balanced.csv', index=False)
val_idx.to_csv('val_idx_balanced.csv', index=False)
test_idx.to_csv('test_idx_balanced.csv', index=False)
balanced_data.to_csv('balanced_data.csv', index=False)


# # 1. PCA

# In[19]:


scaler = StandardScaler()
X = X_rus
X = scaler.fit_transform(X)


# In[24]:


pca_model = PCA(n_components=10)
pca_model.fit(X)


# In[25]:


np.sum(pca_model.explained_variance_ratio_)


# In[26]:


X = pca_model.transform(X)
X = pd.DataFrame(X)
y = y_rus
swarm_pca = pd.concat([X, y], axis=1)
swarm_pca.head()


# In[27]:


swarm_pca.shape


# In[28]:


swarm_pca.to_csv('balanced_pca.csv', index=False)


# # 2. LDA

# In[6]:


data = pd.read_csv('Swarm_Behaviour.csv')
train_idx = pd.read_csv('train_idx.csv')
val_idx = pd.read_csv('val_idx.csv')
test_idx = pd.read_csv('test_idx.csv')
for i in range(10):
    train_index = train_idx[train_idx.columns[i]]
    val_index = val_idx[val_idx.columns[i]]
    test_index = test_idx[test_idx.columns[i]]

    train_index = train_index.values.reshape(-1)
    val_index = val_index.values.reshape(-1)
    test_index = test_index.values.reshape(-1)

    X = data.drop("Swarm_Behaviour",axis=1)
    y = data["Swarm_Behaviour"]

    X_train = X.iloc[train_index]
    X_val = X.iloc[val_index]
    X_test = X.iloc[test_index]

    y_train = y[train_index]
    y_val = y[val_index]
    y_test = y[test_index]

    X_val_test = pd.concat([X_train, X_val], axis=0)
    y_val_test = pd.concat([y_train, y_val], axis=0)

    # LDA
    scaler = StandardScaler()
    lda_model = LDA(n_components=1)
    lda_model.fit(X_train, y_train)
    X = lda_model.transform(X)
    X = pd.DataFrame(X)
    swarm_lda = pd.concat([X, y], axis=1)
    swarm_lda.to_csv('lda_'+str(i)+'.csv', index=False)


# In[7]:


# balanced
data = pd.read_csv('balanced_data.csv')
train_idx = pd.read_csv('train_idx_balanced.csv')
val_idx = pd.read_csv('val_idx_balanced.csv')
test_idx = pd.read_csv('test_idx_balanced.csv')
for i in range(10):
    train_index = train_idx[train_idx.columns[i]]
    val_index = val_idx[val_idx.columns[i]]
    test_index = test_idx[test_idx.columns[i]]

    train_index = train_index.values.reshape(-1)
    val_index = val_index.values.reshape(-1)
    test_index = test_index.values.reshape(-1)

    X = data.drop("Swarm_Behaviour",axis=1)
    y = data["Swarm_Behaviour"]

    X_train = X.iloc[train_index]
    X_val = X.iloc[val_index]
    X_test = X.iloc[test_index]

    y_train = y[train_index]
    y_val = y[val_index]
    y_test = y[test_index]

    X_val_test = pd.concat([X_train, X_val], axis=0)
    y_val_test = pd.concat([y_train, y_val], axis=0)

    # LDA
    scaler = StandardScaler()
    lda_model = LDA(n_components=1)
    lda_model.fit(X_train, y_train)
    X = lda_model.transform(X)
    X = pd.DataFrame(X)
    swarm_lda = pd.concat([X, y], axis=1)
    swarm_lda.to_csv('balanced_lda_'+str(i)+'.csv', index=False)


# # Autoencoder

# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[ ]:


def load_data(data):
    df = pd.read_csv(data)

    y = df["Swarm_Behaviour"]

    scaler = StandardScaler()
    X = df.drop("Swarm_Behaviour",axis=1)
    X = scaler.fit_transform(X)    
    return X, scaler, y

def numpyToTensor(x):
    x_train = torch.from_numpy(x).to(device)
    return x_train


# In[ ]:


class DataBuilder(Dataset):
    def __init__(self, path):
        self.x, self.standardizer, self.y = load_data(path)
        self.x = numpyToTensor(self.x)
        self.len=self.x.shape[0]
    def __getitem__(self,index):      
        return self.x[index]
    def __len__(self):
        return self.len

data_set = DataBuilder('balanced_data.csv')
trainloader = DataLoader(dataset=data_set, batch_size=1024)


# In[ ]:


class Autoencoder(nn.Module):
    """Makes the main denoising auto

    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(in_shape, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, enc_shape),
        )
        
        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, in_shape)
        )
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
encoder = Autoencoder(in_shape=2400, enc_shape=16).double().to(device)

error = nn.MSELoss()

optimizer = optim.Adam(encoder.parameters())


# In[ ]:


def train(model, error, optimizer, n_epochs, x):
    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        output = model(x)
        loss = error(output, x)
        loss.backward()
        optimizer.step()
        
        if epoch % int(0.01*n_epochs) == 0:
            print(f'epoch {epoch} \t Loss: {loss.item():.4g}')


# In[ ]:


train(encoder, error, optimizer, 10000, data_set.x)


# In[ ]:


with torch.no_grad():
    encoded = encoder.encode(data_set.x)
    decoded = encoder.decode(encoded)
    mse = error(decoded, data_set.x).item()
    enc = encoded.cpu().detach().numpy()
    dec = decoded.cpu().detach().numpy()


# In[ ]:


X = pd.DataFrame(enc)
y = pd.DataFrame(data_set.y)
y = y.reset_index(drop=True)
auto_enc = pd.concat([X, y], axis=1)
auto_enc.to_csv('balanced_autoenc.csv', index=False)

