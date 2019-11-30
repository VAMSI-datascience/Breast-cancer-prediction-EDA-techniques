
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import time


# In[2]:


data= pd.read_csv('data.csv')


# In[3]:


data.head(10)


# In[4]:


data.info()


# In[5]:


data['diagnosis']


# In[6]:


dummy = ['diagnosis']


# In[8]:


final = pd.get_dummies(data,columns=dummy,drop_first=True)


# In[9]:


final.head(10)


# In[10]:


final.isnull().sum()


# In[11]:


final.drop(['Unnamed: 32'],axis=1,inplace=True)
final.columns


# In[12]:


final.isnull().sum()


# In[13]:


final.head(10)


# In[14]:


z = final.drop('id',axis=1)


# In[15]:


y = final.diagnosis_M


# In[16]:


final[final['diagnosis_M']==1]['radius_mean'].hist(bins=30,color='red',label='diagnosis_M 1',alpha=0.6)
final[final['diagnosis_M']==0]['radius_mean'].hist(bins=30,color='green',label='diagnosis_M 0',alpha=0.6)

plt.legend()
plt.xlabel("radius_mean")



# In[17]:


final[final['diagnosis_M']==1]['texture_mean'].hist(bins=30,color='red',label='diagnosis_M 1',alpha=0.6)
final[final['diagnosis_M']==0]['texture_mean'].hist(bins=30,color='green',label='diagnosis_M 0',alpha=0.6)

plt.legend()
plt.xlabel("texture_mean")


# In[18]:


final[final['diagnosis_M']==1]['perimeter_mean'].hist(bins=30,color='red',label='diagnosis_M 1',alpha=0.6)
final[final['diagnosis_M']==0]['perimeter_mean'].hist(bins=30,color='green',label='diagnosis_M 0',alpha=0.6)

plt.legend()
plt.xlabel("perimeter_mean")


# In[19]:


final[final['diagnosis_M']==1]['area_mean'].hist(bins=30,color='red',label='diagnosis_M 1',alpha=0.6)
final[final['diagnosis_M']==0]['area_mean'].hist(bins=30,color='green',label='diagnosis_M 0',alpha=0.6)

plt.legend()
plt.xlabel("area_mean")


# In[20]:


final[final['diagnosis_M']==1]['smoothness_mean'].hist(bins=30,color='red',label='diagnosis_M 1',alpha=0.6)
final[final['diagnosis_M']==0]['smoothness_mean'].hist(bins=30,color='green',label='diagnosis_M 0',alpha=0.6)

plt.legend()
plt.xlabel("smoothness_mean")


# In[21]:


final[final['diagnosis_M']==1]['compactness_mean'].hist(bins=30,color='red',label='diagnosis_M 1',alpha=0.6)
final[final['diagnosis_M']==0]['compactness_mean'].hist(bins=30,color='green',label='diagnosis_M 0',alpha=0.6)

plt.legend()
plt.xlabel("compactness_mean")


# In[22]:


final[final['diagnosis_M']==1]['concavity_mean'].hist(bins=30,color='red',label='diagnosis_M 1',alpha=0.6)
final[final['diagnosis_M']==0]['concavity_mean'].hist(bins=30,color='green',label='diagnosis_M 0',alpha=0.6)

plt.legend()
plt.xlabel("concavity_mean")


# In[23]:


final[final['diagnosis_M']==1]['concave points_mean'].hist(bins=30,color='red',label='diagnosis_M 1',alpha=0.6)
final[final['diagnosis_M']==0]['concave points_mean'].hist(bins=30,color='green',label='diagnosis_M 0',alpha=0.6)

plt.legend()
plt.xlabel("concave points_mean")


# In[24]:


final[final['diagnosis_M']==1]['symmetry_mean'].hist(bins=30,color='red',label='diagnosis_M 1',alpha=0.6)
final[final['diagnosis_M']==0]['symmetry_mean'].hist(bins=30,color='green',label='diagnosis_M 0',alpha=0.6)

plt.legend()
plt.xlabel("symmetry_mean")


# In[25]:


final[final['diagnosis_M']==1]['fractal_dimension_mean'].hist(bins=30,color='red',label='diagnosis_M 1',alpha=0.6)
final[final['diagnosis_M']==0]['fractal_dimension_mean'].hist(bins=30,color='green',label='diagnosis_M 0',alpha=0.6)

plt.legend()
plt.xlabel("fractal_dimension_mean")


# In[26]:


sns.set(style="whitegrid", palette="muted")
data_dia = y
data = z
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis_M",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis_M", data=data)

plt.xticks(rotation=90)


# In[27]:


sns.set(style="whitegrid", palette="muted")
data_dia = y
data = z
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,11:21]],axis=1)
data = pd.melt(data,id_vars="diagnosis_M",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis_M", data=data)

plt.xticks(rotation=90)


# In[29]:


sns.set(style="whitegrid", palette="muted")
data_dia = y
data = final
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,21:31]],axis=1)
data = pd.melt(data,id_vars="diagnosis_M",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis_M", data=data)

plt.xticks(rotation=90)


# In[30]:


#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(z.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[31]:


z.columns


# In[109]:


drop_list = ['texture_mean','smoothness_mean','compactness_mean','concavity_mean','symmetry_mean','fractal_dimension_mean',
            'radius_se', 'texture_se','area_se', 'smoothness_se',
       'compactness_se', 'concave points_se', 'symmetry_se','fractal_dimension_se','texture_worst','smoothness_worst','compactness_worst','symmetry_worst', 'fractal_dimension_worst']
r = z.drop(drop_list,axis=1)


# In[110]:


r.head()


# In[111]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix,classification_report
from sklearn.metrics import accuracy_score


# In[166]:


X=r.drop('diagnosis_M',axis=1)
y=r['diagnosis_M']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# In[167]:


model = RandomForestClassifier(n_estimators=300)


# In[168]:


model.fit(X_train,y_train)


# In[169]:


pred = model.predict(X_test)


# In[170]:


print(confusion_matrix(y_test,pred))


# In[171]:


print(accuracy_score(y_test,pred))


# In[172]:


print(classification_report(y_test,pred))

