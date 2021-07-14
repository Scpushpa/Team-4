#!/usr/bin/env python
# coding: utf-8

# In[94]:


# COMP 247 - Project - Group 4
import pandas as pd
import numpy as np


# Reading dataset

# In[95]:


dataset = pd.read_csv("Bicycle_Thefts.csv", header=0, low_memory=False)


# In[96]:


df = pd.DataFrame(dataset)
df


# Dataset features type

# In[97]:


df.dtypes


# In[98]:


# Keys
df.keys()


# Select the only important keys: Month, Area, Cost, and Status of stolen features

# In[99]:


df = df[['Occurrence_Month', 'Division', 'Hood_ID', 'Cost_of_Bike', 'Status']]


# In[100]:


# Rename some column just for easy co
df=df.rename(columns={'Occurrence_Month':'Month', 'Division':'Div', "Hood_ID" : "Hood", "Cost_of_Bike" : "Cost"})
df.head(5)


# Evaluate Month feature

# In[101]:


df.Month.value_counts()


# Evaluate Neigborhood feature

# In[102]:


df.Hood.value_counts()


# Evaluate Division feature

# In[103]:


df.Div.value_counts()


# We have NSA which is unknown location, so we assign '000' to this Value

# In[104]:


# replace NSA in Division
df.replace("NSA", "D00", inplace=True)
df.Div.value_counts()


# Evaluate Status feature

# In[105]:


df.Status.value_counts()


# We have 454 case UNKNOWN that means Lost bike (stolen or not), so, we assign them to STOLEN case

# In[106]:


df.replace("UNKNOWN", "STOLEN", inplace=True)


# In[107]:


# review the target again
df.Status.value_counts()


# Set target to number value: STOLEN=0, RECOVERED=1

# In[108]:


df.replace({'STOLEN': 0, 'RECOVERED': 1}, inplace=True)


# In[109]:


df.Status.value_counts()


# In[110]:


df.dtypes


# In[111]:


df['Hood'] = df['Hood'].astype(int)


# In[112]:


# convert Division number part
df['Div'] = df['Div'].str.slice(1)
df['Div'] = df['Div'].astype(int)


# In[113]:


df.head(5)


# In[114]:


def monthN(s):
    switcher = {
        'January':      1,
        'February':     2,
        'March':        3,
        'April':        4,
        'May':          5,
        'June':         6,
        'July':         7,
        'August':       8,
        'September':    9,
        'October':      10,
        'November':     11,
        'December':     12
    }
    return switcher.get(s, "Invalid month")


# In[115]:


df['Month'] = df['Month'].apply(lambda x: monthN(x))


# In[116]:


df.head(5)


# In[117]:


df.isnull().sum()


# In[118]:


df['Cost'].describe()


# In[119]:


df['Cost'].replace({0: np.nan}, inplace=True)


# In[120]:


df['Cost'].describe()


# In[126]:


df.Cost.fillna(df["Cost"].median(), inplace = True)


# In[127]:


df


# In[128]:


X = df[['Month', 'Div', 'Hood', 'Cost']]
y = df[['Status']]


# In[130]:


X.head(5), y.head(5)


# In[ ]:




