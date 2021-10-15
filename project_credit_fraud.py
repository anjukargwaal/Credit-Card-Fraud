#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import numpy as np
import pandas as pd
import seaborn 
import scipy
import sklearn
import matplotlib.pyplot as plt


# In[5]:


df=pd.read_csv("creditcard.csv")
df.head()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.columns


# In[12]:


df.hist(figsize=(20,20))
plt.show()


# In[13]:


fraud=df[df["Class"]==1]
valid=df[df["Class"]==0]
print(len(fraud))
print(len(valid))


# In[15]:


corrmat=df.corr()

fig=plt.figure(figsize=(12,9))
seaborn.heatmap(corrmat, vmax=.8, square= True)
plt.show()


# In[16]:


Y=df['Class']
X=df.drop('Class',axis=1)

print(X.shape)
print(Y.shape)


# In[17]:


df.isnull().sum()


# In[18]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


# In[20]:


xtrain,xtest,ytrain,ytest=sklearn.model_selection.train_test_split(X,Y,test_size=0.20,random_state=5)


# In[21]:



lm=LogisticRegression()


# In[22]:


lm.fit(xtrain,ytrain)


# In[23]:



pred=lm.predict(xtest)


# In[24]:


from sklearn.metrics import roc_curve, auc, roc_auc_score
print (roc_auc_score( ytest, pred))


# In[25]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(ytest, pred)
print (auc(false_positive_rate, true_positive_rate))


# In[26]:


from sklearn.tree import DecisionTreeClassifier


# In[27]:


from sklearn.model_selection import RandomizedSearchCV
classifier=DecisionTreeClassifier(criterion='entropy',random_state=5)
classifier.fit(xtrain,ytrain)


# In[28]:


y_pred=classifier.predict(xtest)


# In[29]:


print (roc_auc_score( ytest, y_pred))


# In[30]:


params={"max_depth":[3,None],"min_samples_leaf":[1,9],"criterion":["gini","entropy"]}


# In[31]:


tree=DecisionTreeClassifier()


# In[32]:


tree_cv=RandomizedSearchCV(tree,params,cv=5)


# In[33]:


tree_cv.fit(xtrain,ytrain)


# In[34]:


print("tuned tree params:{}".format(tree_cv.best_params_))
print("best score is:{}".format(tree_cv.best_score_))


# In[35]:


y_pred=tree_cv.predict(xtest)


# In[36]:


print (roc_auc_score( ytest, y_pred))


# In[37]:


from sklearn.model_selection import cross_val_score
tree=DecisionTreeClassifier()
print(cross_val_score(tree,xtrain,ytrain,cv=5,scoring='accuracy'))


# In[38]:


from sklearn.model_selection import cross_val_score
lm=LogisticRegression()
print(cross_val_score(lm,xtrain,ytrain,cv=5,scoring='accuracy'))


# In[39]:


from sklearn.ensemble import RandomForestClassifier
params={"max_depth":[3,None],"min_samples_leaf":[1,9],"criterion":["gini","entropy"]}


# In[40]:



from sklearn.model_selection import RandomizedSearchCV
rf_tree=RandomForestClassifier()
tree_cv=RandomizedSearchCV(rf_tree,params,cv=5)


# In[41]:


search_fit=tree_cv.fit(xtrain,ytrain)


# In[42]:


print("tuned tree params:{}".format(tree_cv.best_params_))
print("best score is:{}".format(tree_cv.best_score_))
best_clf=search_fit.best_estimator_
best_clf


# In[ ]:


from sklearn.model_selection import cross_val_score

print(cross_val_score(best_clf,xtrain,ytrain,cv=5,scoring='accuracy'))


# In[ ]:


y_pred=best_clf.predict(xtest)


# In[ ]:


from sklearn.metrics import roc_curve, auc, roc_auc_score
print (roc_auc_score( ytest, y_pred))


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost


# In[ ]:


params={"learning_rate":[0.05,0.10,0.15,0.20,0.25,0.30],
        "max_depth":[3,4,5,6,8,10,12,None],
        "min_child_weight":[1,3,5,7],
        "gamma":[0.0,0.1,0.2,0.3,0.4],
        "colsample_bytree":[0.3,0.4,0.5,0.7]
       }


# In[ ]:


classifier=xgboost.XGBClassifier()


# In[ ]:


random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring="roc_auc",n_jobs=-1,cv=5,verbose=3)


# In[ ]:


random_search.fit(X,Y)


# In[ ]:


random_search.best_estimator_


# In[ ]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,Y,cv=5)


# In[ ]:


score.mean()


# In[ ]:




