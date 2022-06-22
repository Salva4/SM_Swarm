#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample
import time

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score,roc_auc_score,accuracy_score,classification_report,plot_confusion_matrix,plot_roc_curve,precision_recall_curve


# In[2]:


data = pd.read_csv('Swarm_Behaviour.csv')
data.head()


# In[3]:


train_idx = pd.read_csv('data/train_index.csv')
val_idx = pd.read_csv('data/val_index.csv')
test_idx = pd.read_csv('data/test_index.csv')


# In[4]:


train_idx = train_idx.values.reshape(-1)
val_idx = val_idx.values.reshape(-1)
test_idx = test_idx.values.reshape(-1)


# ## 1. PCA

# In[5]:


data = pd.read_csv('data/swarm_pca3.csv')

X = data.drop("Swarm_Behaviour",axis=1)
y = data["Swarm_Behaviour"]


# In[6]:


label0 = data[data["Swarm_Behaviour"]==0]
label1 = data[data["Swarm_Behaviour"]==1]


# In[7]:


from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_xlabel('first')
ax.set_ylabel('second')
ax.set_zlabel('third')
ax.scatter3D(label0["0"], label0["1"], label0["2"], label = "0")
ax.scatter3D(label1["0"], label1["1"], label1["2"], label = "1")
ax.legend()


# In[8]:


plt.plot(label0["0"], label0["1"],'.', label = "0")
plt.plot(label1["0"], label1["1"],'.', label = "1")
plt.xlabel('first')
plt.ylabel('second')
plt.legend()
plt.show()


# # 2. LDA

# In[9]:


data = pd.read_csv('data/swarm_lda.csv')

X = data.drop("Swarm_Behaviour",axis=1)
y = data["Swarm_Behaviour"]


# In[10]:


label0 = data[data["Swarm_Behaviour"]==0]
label1 = data[data["Swarm_Behaviour"]==1]

plt.plot(label0["0"],'.',label=0)
plt.plot(label1["0"],'.',label=1)
plt.legend()
plt.plot()


# # Experiments ( x 10)

# In[11]:


params_rf={
    "n_estimators":[50,100,150,200,250],
    "max_features":[2,3,4,5,6],
    "max_depth":[4,8,12,16,20]}

params_rf_lda={
    "n_estimators":[50,100,150,200,250],
    "max_depth":[4,8,12,16,20]}

params_xg={
    "n_estimators":[5,10,15,20],
    "learning_rate":[0.05,0.1,0.2],
    "max_depth":[3,4,5,6],
    "subsample":[0.3,0.6,0.9]
}


# In[12]:


def training(data):

    auc_1 = []
    f1_1 = []
    auc_2 = []
    f1_2 = []
    times1 = []
    times2 = []
    
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

        X_train = pd.concat([X_train, X_val], axis=0)
        y_train = pd.concat([y_train, y_val], axis=0)
        
        
        if i == 0:
            # random forest
            rfc = RandomForestClassifier()
            grid_model=GridSearchCV(rfc, params_rf)
            grid_model.fit(X_train,y_train)
            best_params1 = grid_model.best_params_

            # hyperparameters
            n_depth1 = best_params1['max_depth']
            n_features = best_params1['max_features']
            n_estimators1 = best_params1['n_estimators']

            # xgboost
            xgbc = XGBClassifier()
            grid_model=GridSearchCV(xgbc,params_xg)
            grid_model.fit(X_train,y_train)
            best_params2 = grid_model.best_params_

            # hyperparameters
            l_rate = best_params2['learning_rate']
            n_depth2 = best_params2['max_depth']
            n_estimators2 = best_params2['n_estimators']
            subsample = best_params2['subsample']
        
        # random forest
        start1 = time.time()
        rfc = RandomForestClassifier(max_depth = n_depth1,
                                     max_features = n_features,
                                     n_estimators = n_estimators1)
        rfc.fit(X_train, y_train)
        y_predict = rfc.predict(X_test)
        
        auc_1.append(roc_auc_score(y_test, y_predict))
        f1_1.append(f1_score(y_test, y_predict))
        end1 = time.time() 
        times1.append(end1-start1)
        
        # xgboost   
        start2 = time.time()  
        xgbc = XGBClassifier(learning_rate = l_rate, 
                             max_depth = n_depth2, 
                             n_estimators = n_estimators2, 
                             subsample = subsample)
        xgbc.fit(X_train, y_train)
        y_predict = xgbc.predict(X_test)
        roc_auc_score(y_test, y_predict)
        
        auc_2.append(roc_auc_score(y_test, y_predict))
        f1_2.append(f1_score(y_test, y_predict))
        end2 = time.time()
        times2.append(end2-start2)
    
    print(best_params1)
    print("Average AUC score (Random Forest): "+str(sum(auc_1)/10))
    print("Average F1 score (Random Forest): "+str(sum(f1_1)/10))
    print("AUC1: "+str(auc_1))
    print("Time (Random Forest): "+str(times1))
    print("================================================")
    print(best_params2)
    print("Average AUC score (XgBoost): "+str(sum(auc_2)/10))
    print("Average F1 score (XgBoost): "+str(sum(f1_2)/10))
    print("AUC2: "+str(auc_2))
    print("Time (XgBoost): "+str(times2))


# ## PCA

# In[13]:


train_idx = pd.read_csv('train_idx.csv')
val_idx = pd.read_csv('val_idx.csv')
test_idx = pd.read_csv('test_idx.csv')
data = pd.read_csv('data/swarm_pca3.csv')
training(data)


# ## LDA

# In[14]:


def training_lda():

    auc_1 = []
    f1_1 = []
    auc_2 = []
    f1_2 = []
    times1 = []
    times2 = []
    
    for i in range(10):
        data = pd.read_csv('lda_'+str(i)+'.csv')
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

        X_train = pd.concat([X_train, X_val], axis=0)
        y_train = pd.concat([y_train, y_val], axis=0)
        
        if i == 0:
            # random forest
            rfc = RandomForestClassifier()
            grid_model=GridSearchCV(rfc, params_rf_lda)
            grid_model.fit(X_train,y_train)
            best_params1 = grid_model.best_params_

            # hyperparameters
            n_depth1 = best_params1['max_depth']
            n_estimators1 = best_params1['n_estimators']

            # xgboost
            xgbc = XGBClassifier()
            grid_model=GridSearchCV(xgbc,params_xg)
            grid_model.fit(X_train,y_train)
            best_params2 = grid_model.best_params_

            # hyperparameters
            l_rate = best_params2['learning_rate']
            n_depth2 = best_params2['max_depth']
            n_estimators2 = best_params2['n_estimators']
            subsample = best_params2['subsample']
        
        # random forest
        start1 = time.time()
        rfc = RandomForestClassifier(max_depth = n_depth1,
                                     n_estimators = n_estimators1)
        rfc.fit(X_train, y_train)
        y_predict = rfc.predict(X_test)
        
        auc_1.append(roc_auc_score(y_test, y_predict))
        f1_1.append(f1_score(y_test, y_predict))
        end1 = time.time() 
        times1.append(end1-start1)
        
        # xgboost   
        start2 = time.time()  
        xgbc = XGBClassifier(learning_rate = l_rate, 
                             max_depth = n_depth2, 
                             n_estimators = n_estimators2, 
                             subsample = subsample)
        xgbc.fit(X_train, y_train)
        y_predict = xgbc.predict(X_test)
        roc_auc_score(y_test, y_predict)
        
        auc_2.append(roc_auc_score(y_test, y_predict))
        f1_2.append(f1_score(y_test, y_predict))
        end2 = time.time()
        times2.append(end2-start2)
    
    print(best_params1)
    print("Average AUC score (Random Forest): "+str(sum(auc_1)/10))
    print("Average F1 score (Random Forest): "+str(sum(f1_1)/10))
    print("AUC1: "+str(auc_1))
    print("Time (Random Forest): "+str(times1))
    print("================================================")
    print(best_params2)
    print("Average AUC score (XgBoost): "+str(sum(auc_2)/10))
    print("Average F1 score (XgBoost): "+str(sum(f1_2)/10))
    print("AUC2: "+str(auc_2))
    print("Time (XgBoost): "+str(times2))


# In[15]:


train_idx = pd.read_csv('train_idx.csv')
val_idx = pd.read_csv('val_idx.csv')
test_idx = pd.read_csv('test_idx.csv')
training_lda()


# ## Auto-Encoder

# In[16]:


data = pd.read_csv('data/auto_enc.csv')
training(data)


# # Corrupted Data

# In[17]:


def training_corrupted(data, p):
    
    auc_1 = []
    f1_1 = []
    auc_2 = []
    f1_2 = []
    times1 = []
    times2 = []

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

        X_train = pd.concat([X_train, X_val], axis=0)
        y_train = pd.concat([y_train, y_val], axis=0)
        
        n = len(y_train)
        idx = list(y_train.index)
        corrupt_idx = sample(idx, round(n*p))
        
        for j in corrupt_idx:
            y_train[j] = 1 if y_train[j]==0 else 0
        
        if i == 0:
            # random forest
            rfc = RandomForestClassifier()
            grid_model=GridSearchCV(rfc, params_rf)
            grid_model.fit(X_train,y_train)
            best_params1 = grid_model.best_params_

            # hyperparameters
            n_depth1 = best_params1['max_depth']
            n_features = best_params1['max_features']
            n_estimators1 = best_params1['n_estimators']

            # xgboost
            xgbc = XGBClassifier()
            grid_model=GridSearchCV(xgbc,params_xg)
            grid_model.fit(X_train,y_train)
            best_params2 = grid_model.best_params_

            # hyperparameters
            l_rate = best_params2['learning_rate']
            n_depth2 = best_params2['max_depth']
            n_estimators2 = best_params2['n_estimators']
            subsample = best_params2['subsample']
        
        # random forest
        start1 = time.time()
        rfc = RandomForestClassifier(max_depth = n_depth1,
                                     max_features = n_features,
                                     n_estimators = n_estimators1)
        rfc.fit(X_train, y_train)
        y_predict = rfc.predict(X_test)
        
        auc_1.append(roc_auc_score(y_test, y_predict))
        f1_1.append(f1_score(y_test, y_predict))
        end1 = time.time() 
        times1.append(end1-start1)
        
        # xgboost   
        start2 = time.time()  
        xgbc = XGBClassifier(learning_rate = l_rate, 
                             max_depth = n_depth2, 
                             n_estimators = n_estimators2, 
                             subsample = subsample)
        xgbc.fit(X_train, y_train)
        y_predict = xgbc.predict(X_test)
        roc_auc_score(y_test, y_predict)
        
        auc_2.append(roc_auc_score(y_test, y_predict))
        f1_2.append(f1_score(y_test, y_predict))
        end2 = time.time()
        times2.append(end2-start2)
    
    print(best_params1)
    print("Average AUC score (Random Forest): "+str(sum(auc_1)/10))
    print("Average F1 score (Random Forest): "+str(sum(f1_1)/10))
    print("AUC1: "+str(auc_1))
    print("Time (Random Forest): "+str(times1))
    print("================================================")
    print(best_params2)
    print("Average AUC score (XgBoost): "+str(sum(auc_2)/10))
    print("Average F1 score (XgBoost): "+str(sum(f1_2)/10))
    print("AUC2: "+str(auc_2))
    print("Time (XgBoost): "+str(times2))


# In[18]:


data = pd.read_csv('data/swarm_pca3.csv')
train_idx = pd.read_csv('train_idx.csv')
val_idx = pd.read_csv('val_idx.csv')
test_idx = pd.read_csv('test_idx.csv')


# ## 1% Corruption

# In[19]:


training_corrupted(data, 0.01)


# ## 2% Corruption

# In[20]:


training_corrupted(data, 0.02)


# ## 3% Corruption

# In[21]:


training_corrupted(data, 0.03)


# ## Reducing the sample size

# In[22]:


data = pd.read_csv('data/swarm_pca3.csv')
train_idx = pd.read_csv('train_idx_small.csv')
val_idx = pd.read_csv('val_idx_small.csv')
test_idx = pd.read_csv('test_idx_small.csv')


# ## 1% Corruption

# In[23]:


training_corrupted(data, 0.01)


# ## 2% Corruption

# In[24]:


training_corrupted(data, 0.02)


# ## 3% Corruption

# In[25]:


training_corrupted(data, 0.03)


# ## PCA-10 SMALL (uncorrupted)

# In[26]:


data = pd.read_csv('data/swarm_pca3.csv')
train_idx = pd.read_csv('train_idx_small.csv')
val_idx = pd.read_csv('val_idx_small.csv')
test_idx = pd.read_csv('test_idx_small.csv')
training(data)


# ## Balanced Data

# In[27]:


data = pd.read_csv('balanced_data.csv')
train_idx = pd.read_csv('train_idx_balanced.csv')
val_idx = pd.read_csv('val_idx_balanced.csv')
test_idx = pd.read_csv('test_idx_balanced.csv')
training(data)


# ### PCA

# In[28]:


data = pd.read_csv('balanced_pca.csv')
training(data)


# ### LDA

# In[29]:


def training_lda_balanced():

    auc_1 = []
    f1_1 = []
    auc_2 = []
    f1_2 = []
    times1 = []
    times2 = []
    
    for i in range(10):
        data = pd.read_csv('balanced_lda_'+str(i)+'.csv')
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

        X_train = pd.concat([X_train, X_val], axis=0)
        y_train = pd.concat([y_train, y_val], axis=0)
        
        if i == 0:
            # random forest
            rfc = RandomForestClassifier()
            grid_model=GridSearchCV(rfc, params_rf_lda)
            grid_model.fit(X_train,y_train)
            best_params1 = grid_model.best_params_

            # hyperparameters
            n_depth1 = best_params1['max_depth']
            n_estimators1 = best_params1['n_estimators']

            # xgboost
            xgbc = XGBClassifier()
            grid_model=GridSearchCV(xgbc,params_xg)
            grid_model.fit(X_train,y_train)
            best_params2 = grid_model.best_params_

            # hyperparameters
            l_rate = best_params2['learning_rate']
            n_depth2 = best_params2['max_depth']
            n_estimators2 = best_params2['n_estimators']
            subsample = best_params2['subsample']
        
        # random forest
        start1 = time.time()
        rfc = RandomForestClassifier(max_depth = n_depth1,
                                     n_estimators = n_estimators1)
        rfc.fit(X_train, y_train)
        y_predict = rfc.predict(X_test)
        
        auc_1.append(roc_auc_score(y_test, y_predict))
        f1_1.append(f1_score(y_test, y_predict))
        end1 = time.time() 
        times1.append(end1-start1)
        
        # xgboost   
        start2 = time.time()  
        xgbc = XGBClassifier(learning_rate = l_rate, 
                             max_depth = n_depth2, 
                             n_estimators = n_estimators2, 
                             subsample = subsample)
        xgbc.fit(X_train, y_train)
        y_predict = xgbc.predict(X_test)
        roc_auc_score(y_test, y_predict)
        
        auc_2.append(roc_auc_score(y_test, y_predict))
        f1_2.append(f1_score(y_test, y_predict))
        end2 = time.time()
        times2.append(end2-start2)
    
    print(best_params1)
    print("Average AUC score (Random Forest): "+str(sum(auc_1)/10))
    print("Average F1 score (Random Forest): "+str(sum(f1_1)/10))
    print("AUC1: "+str(auc_1))
    print("Time (Random Forest): "+str(times1))
    print("================================================")
    print(best_params2)
    print("Average AUC score (XgBoost): "+str(sum(auc_2)/10))
    print("Average F1 score (XgBoost): "+str(sum(f1_2)/10))
    print("AUC2: "+str(auc_2))
    print("Time (XgBoost): "+str(times2))
    


# In[30]:


train_idx = pd.read_csv('train_idx_balanced.csv')
val_idx = pd.read_csv('val_idx_balanced.csv')
test_idx = pd.read_csv('test_idx_balanced.csv')
training_lda_balanced()


# ### Autoencoder

# In[31]:


data = pd.read_csv('balanced_autoenc.csv')
training(data)


# ## Original Data

# In[32]:


data = pd.read_csv('Swarm_Behaviour.csv')
training(data)

