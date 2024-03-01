#!/usr/bin/env python
# coding: utf-8

# In[217]:


import pandas as pd 


# In[218]:


teams = pd.read_csv('/users/dr3346/downloads/teams.csv')


# In[219]:


# Remove some columns 
#you use double brackets when you want to select multiple columns, and single brackets when you want to select a single column
teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]


# In[220]:


teams


# In[221]:


# correlation between medals and all other variables 
#To silence the warning, you can explicitly set numeric_only=True
teams.corr(numeric_only=True)["medals"]


# In[222]:


import seaborn as sns


# In[223]:


#lmplot() --> to create a scatter plot with a linear regression line.
#fit_reg = True --> will fit a linear regression model to the data and plot the regression line.
sns.lmplot(x="athletes", y="medals", data = teams, fit_reg = True)


# In[224]:


sns.lmplot(x="age", y="medals", data = teams, fit_reg = True)


# In[225]:


teams.plot.hist(y="medals")


# In[226]:


# any(axis=1)--> returns a missing value for any attribute in a row 
teams[teams.isnull().any(axis=1)]


# In[227]:


teams = teams.dropna()


# In[228]:


# Take last 2 years of data and place them in our test data set
train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()


# In[229]:


train.shape  


# In[230]:


test.shape  


# In[231]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression()


# In[232]:


# Train LR lodel to use features to guess target variable
predictors = ["athletes", "prev_medals"] #features
target = "medals" #targetvariable


# In[233]:


# Call fit method to fit linear regression model. Pass in predictors from testing set, insert target
reg.fit(train[predictors], train["medals"])


# In[234]:


# Only pass thru predictors. We generate predictions. Shows in a numpy array
predictions = reg.predict(test[predictors])


# In[235]:


#Assign predictions to a column
test["predictions"] = predictions


# In[236]:


test


# In[237]:


# if preditions are less than 0, they are turned into 0
test.loc[test["predictions"] < 0 , "predictions"] = 0 


# In[238]:


test["predictions"] = test["predictions"].round()


# In[239]:


test


# In[240]:


from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(test["medals"], test["predictions"])
error


# In[241]:


# Is error below Standard Deviation?
teams.describe()["medals"]


# In[242]:


test["predictions"] = predictions


# In[243]:


test[test["team"] == 'USA']


# In[244]:


test[test["team"] == 'IND']


# In[245]:


errors = (test["medals"] - predictions).abs()


# In[246]:


test[test["team"] == 'USA']


# In[247]:


error_by_team = errors.groupby(test["team"]).mean()
medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio =  error_by_team / medals_by_team 


# In[248]:


error_by_team


# In[249]:


import numpy as np
error_ratio = error_ratio[np.isfinite(error_ratio)]


# In[250]:


error_ratio


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




