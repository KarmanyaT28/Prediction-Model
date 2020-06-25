#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[20]:


dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[22]:


print(X_train)


# In[23]:


print(y_train)


# In[24]:


print(X_test)


# In[25]:


print(y_test)


# In[26]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[27]:


print(X_train)


# In[28]:


print(X_test)


# In[29]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[30]:


y_pred = classifier.predict(X_test)


# In[31]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[32]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[ ]:




