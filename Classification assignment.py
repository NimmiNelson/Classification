#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np                        #importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_excel(r"C:\Users\DELL\Downloads\iris (1).xls")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.isna().sum()


# In[9]:


df['Classification'].unique()


# In[10]:


df['Classification'].value_counts()


# In[11]:


df['Classification'].value_counts(normalize=True)


# In[13]:


for i in['SL','SW','PL']:
    df[i]=df[i].fillna(df[i].median())
         


# In[14]:


df.isna().sum()


# In[15]:


df.shape


# In[16]:


data=pd.get_dummies(df)


# In[17]:


data.head()


# In[21]:


data.tail()


# In[23]:


data['Classification_Iris-setosa'].nunique()


# In[18]:


data.shape


# In[20]:


corrmatrix=data.corr()             #Finding the correlation between the 2 variables.
sns.heatmap(corrmatrix,annot=True)


# In[24]:


y=data['Classification_Iris-virginica']                           
x=data.drop(['Classification_Iris-virginica'],axis=1)    


# In[25]:


x.head()


# In[26]:


y.head()


# In[27]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=41)


# In[28]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logit_model=LogisticRegression()
logit_model.fit(x_train,y_train)
y_pred_lr=logit_model.predict(x_test)


# In[32]:


y_test,y_pred_lr


# In[29]:


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score


# In[31]:


print("Accuracy =",accuracy_score(y_test,y_pred_lr))


# In[33]:


print("precision =",precision_score(y_test,y_pred_lr))
print("recall =",recall_score(y_test,y_pred_lr))
print("f1-score =",f1_score(y_test,y_pred_lr))


# In[34]:


confusion_matrix(y_test,y_pred_lr)


# In[35]:


y_pred_lr


# In[41]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
metric=[]
neigbors=np.arange(3,15)
for k in neigbors:
    classifier=KNeighborsClassifier(n_neighbors=k,metric="minkowski",p=2)
    classifier.fit(x_train,y_train)
    y_pred_knn=classifier.predict(x_test)
    acc=accuracy_score(y_test,y_pred_knn)
    metric.append(acc)
    


# In[42]:


plt.plot(neigbors,metric,'o-')
plt.xlabel('k value')
plt.ylabel('accuracy')
plt.grid()


# In[ ]:




