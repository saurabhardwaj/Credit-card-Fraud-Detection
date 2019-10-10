#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np


# In[2]:


os.getcwd()


# In[3]:


os.chdir('C:\\Users\\bhardwaj\\Desktop')


# In[4]:


credit = pd.read_csv("creditcard.csv")


# In[5]:


credit.head()


# In[6]:


credit['Class'].value_counts()


# In[7]:


# Totol amount froud detected
credit.groupby('Class')[['Amount']].sum()


# In[8]:


credit.groupby('Class')[['Amount']].max()


# In[9]:


credit.groupby('Class')[['Amount']].mean()


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


import seaborn as sns


# In[12]:


credit.hist(column='Amount')


# In[13]:


plt.plot(credit.Amount,color='blue')
plt.plot(credit.Class,color='red')
plt.xlabel('Amount')
plt.ylabel('Class')
plt.show()


# In[14]:


credit.shape


# In[15]:


# fraud percentage 
percentage=492*100/284807


# In[16]:


print(percentage)


# In[17]:


credit1 = credit.drop(['Class'],axis =1)


# In[18]:


credit2 = credit['Class']


# In[19]:


credit2.head()


# In[20]:


credit.dtypes


# In[21]:


from sklearn import linear_model
from sklearn import metrics


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


# Defining the X and y variables 
#splitting te data
X = credit1
y = credit2
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)


# In[24]:


#creating logistic regression 
reg = linear_model.LogisticRegression()


# In[25]:


# train the model on training set
mod = reg.fit(X_train,y_train)


# In[26]:


# making prediction on the test set
pred = reg.predict(X_test)


# In[27]:


#accuracy
print('accuracy:',metrics.accuracy_score(y_test,pred)*100)


# In[28]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[29]:


from sklearn.metrics import confusion_matrix
confusion_matrix(
    y_test.values, pred)


# In[30]:


confusion_score = (85278+80)/(85278+30+55+80)


# In[31]:


print(confusion_score)


# In[32]:


credit.isnull().sum()


# In[33]:


from sklearn.tree import DecisionTreeClassifier


# In[34]:


dit = DecisionTreeClassifier()
mod1 = dit.fit(X_train,y_train)
Y_pred = dit.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test,Y_pred))


# In[35]:


det1 = DecisionTreeClassifier(criterion ='entropy',max_depth = 3,random_state = 1)


# In[36]:


mod2 = det1.fit(X_train,y_train)
prew = det1.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test,prew))


# In[37]:


get_ipython().system('pip install graphviz')


# In[38]:


get_ipython().system('pip install pydotplus')


# In[39]:


import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38/bin/'


# In[40]:


list(credit1.columns)


# In[41]:


feature_col = ['Time',
 'V1',
 'V2',
 'V3',
 'V4',
 'V5',
 'V6',
 'V7',
 'V8',
 'V9',
 'V10',
 'V11',
 'V12',
 'V13',
 'V14',
 'V15',
 'V16',
 'V17',
 'V18',
 'V19',
 'V20',
 'V21',
 'V22',
 'V23',
 'V24',
 'V25',
 'V26',
 'V27',
 'V28',
 'Amount']


# In[42]:


from IPython.display import Image  
from sklearn import tree
import pydotplus # installing pyparsing maybe needed
...

dot_data = tree.export_graphviz(det1, out_file=None,feature_names = feature_col)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


# In[43]:


import seaborn as sns
corrmatrix = credit.corr()
f, ax = plt.subplots(figsize=(10,7))
sns.heatmap(corrmatrix, vmax=0.8, square=True)


# In[44]:


det3 = DecisionTreeClassifier(criterion ='gini',max_depth = 3,random_state = 1)


# In[45]:


mod3 = det3.fit(X_train,y_train)
pre = det1.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test,pre))


# In[46]:


from IPython.display import Image  
from sklearn import tree
import pydotplus # installing pyparsing maybe needed
...

dot_data = tree.export_graphviz(det3, out_file=None,feature_names = feature_col)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


# In[47]:


from sklearn import metrics


# In[48]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=None)


# In[49]:


from sklearn.ensemble import RandomForestClassifier


# In[50]:


treeclass = RandomForestClassifier(n_estimators = 1000)


# In[51]:


treeclass.fit(X_train,y_train)


# In[53]:


y_pred  = treeclass.predict(X_test)


# In[54]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[55]:


clf = tree.DecisionTreeClassifier(max_depth=3,random_state=200)


# In[58]:


import sklearn.model_selection as model_selection


# In[59]:


mod = model_selection.GridSearchCV(clf,param_grid={'max_depth':[2,3,4,5,6,]})
mod.fit(X_train,y_train)


# In[60]:


mod.best_estimator_


# In[61]:


mod.best_score_


# In[ ]:




