#!/usr/bin/env python
# coding: utf-8

# ## To predict the Gender by First Name of person

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


col_name = ['First_Name', 'Gender', 'Value']                             ## Defining the column names
data = pd.read_csv('name_gender.csv', names = col_name, header = None)   ## Reading the dataset with defined column_names


# In[3]:


data.head()


# In[4]:


data.shape   ## Dimansion of the dataset


# In[5]:


sns.countplot(data.Gender)      ## Counts of classes in gender


# In[6]:


data.Gender.value_counts()     ## Showing counts of each Classes in Gender


# In[7]:


len(data.First_Name.unique())


# #### From the above we see that all first names are unique names

# In[8]:


data.Gender.replace({'F':1, 'M': 0}, inplace = True)    ## Replacing the values in Gender with numbers


# In[9]:


data.head(3)


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[32]:


tfidf_vec = TfidfVectorizer(analyzer='char')       


# In[33]:


X = data['First_Name']
X_new = tfidf_vec.fit(X)              ## Using Count_Vectorizer to initialize the term frequencies by characters of names


# In[34]:


len(X_new.vocabulary_)                ## Vocab of 26 characters


# In[35]:


X_matrix = X_new.transform(X)


# In[36]:


data['First_Name'][0]


# In[37]:


X_matrix[0,:].toarray()


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


## Splitting data for validation

x_train, x_test, y_train, y_test = train_test_split(X_matrix, data.Gender, test_size = 0.3, random_state = 42)


# In[40]:


from keras.utils import to_categorical


# In[41]:


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[42]:


## Loading the model for validating

import pickle

loaded_model = pickle.load(open('model1.sav', 'rb'))


# In[43]:


prediction = loaded_model.predict_classes(x_test)


# In[44]:


from sklearn import metrics


# In[45]:


metrics.confusion_matrix(y_test[:,1], prediction)


# In[46]:


acc = metrics.accuracy_score(y_test[:,1], prediction)
print(f'Accuracy: {(acc * 100) : .2f}%')           ## Using format specifier to print accuracy to 2 decimals.


# In[ ]:




