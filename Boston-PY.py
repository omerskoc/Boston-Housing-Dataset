#!/usr/bin/env python
# coding: utf-8

# # Data Exploration

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#read in the data using pandas
boston_data=pd.read_csv("Boston_Data.csv")
boston_data.head()


# In[3]:


boston_data.describe()


# In[4]:


boston_data.shape


# In[5]:


# Print column names and type
boston_data.info()


# In[6]:


# check null for each features
boston_data.isnull().sum()


# In[7]:


X = boston_data.drop('crim', axis = 1 )
Y = boston_data['crim']


# In[8]:


from sklearn.model_selection import train_test_split
#split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=5)


# In[9]:


x_train.shape


# In[10]:


x_test.shape


# In[11]:


# As response variables are continious, we should apply regression.
Y.head()


# # Multiple Linear Regression

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


#read in the data using pandas
data=pd.read_csv("Boston_Data.csv")
X = boston_data.drop('crim', axis = 1 )
Y = boston_data['crim']


# In[14]:


from sklearn.model_selection import train_test_split
#split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=5)


# In[15]:


from sklearn.linear_model import LinearRegression
# Create Linear Regression 
lm = LinearRegression()
# Fit the the data
lm.fit(x_train, y_train)


# In[16]:


#Make prediction using test data
y_pred = lm.predict(x_test)
y_pred


# In[17]:


# Use score method to get test accuracy of model
score = lm.score(x_test, y_test)
print(score)


# In[18]:


pd.DataFrame(zip(X.columns, lm.coef_),
columns=['features', 'estimatedCoeffs'])


# In[19]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test,y_pred)
print('Mean Squared Error: ',format(mse,'.4f'))
mae = mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error: ',format(mae,'.4f'))
rsq = r2_score(y_test,y_pred) #R-Squared on the testing data
print('R-square: ',format(rsq,'.4f'))


# # OLS

# In[20]:


boston_data=pd.read_csv("Boston_Data.csv")

X = boston_data.drop('crim', axis = 1 )
Y = boston_data['crim']


# In[21]:


from sklearn.model_selection import train_test_split
#split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=5)


# In[22]:


import statsmodels.api as sm
X= sm.add_constant(X)
lm = sm.OLS(y_train, x_train)
model = lm.fit()
model.summary()


# In[23]:


import statistics
y_pred=model.predict(x_test)
y_pred


# In[24]:


y_test


# In[25]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test,y_pred)
print('Mean Squared Error: ',format(mse,'.4f'))
mae = mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error: ',format(mae,'.4f'))
rsq = r2_score(y_test,y_pred) #R-Squared on the testing data
print('R-square: ',format(rsq,'.4f'))


# # Best Subset

# In[26]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(0)
boston_data=pd.read_csv("Boston_Data.csv")


# In[27]:


X = boston_data.drop('crim', axis = 1 )
Y = boston_data['crim']


# In[28]:


from sklearn.model_selection import train_test_split
#split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=5)


# In[29]:


from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
# Create Linear Regression classifier
lr = LinearRegression()
efs = EFS(lr, 
          min_features=1,
          max_features=13,
          scoring='neg_mean_squared_error',
          print_progress=True,
          cv=5)

feature_names = ('zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat','medv')
efs.fit(x_train, y_train, custom_feature_names = feature_names)

print('Best score: %.2f' % (efs.best_score_*(-1)))
print('Best subset (indices):', efs.best_idx_)
print('Best subset (corresponding names):', efs.best_feature_names_)


# In[30]:


pd.DataFrame.from_dict(efs.get_metric_dict()).T.sort_values('avg_score',ascending=False)


# In[31]:


x_train_selected = x_train.iloc[:,[0, 1, 3, 6, 7, 9, 12]]
x_test_selected = x_test.iloc[:,[0, 1, 3, 6, 7, 9, 12]]
lr.fit(x_train_selected,y_train)

pred = lr.predict(x_test_selected)
mse=mean_squared_error(y_test,pred)
print('MSE: ', format(mse,'.2f'))
rsq = r2_score(y_test,pred)
print('R-square: ',format(rsq,'.4f'))


# # Forward Selection

# In[32]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

np.random.seed(0)
boston_data=pd.read_csv("Boston_Data.csv")

X = boston_data.drop('crim', axis = 1 )
Y = boston_data['crim']

#split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=5)


# In[33]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
lr = LinearRegression()

sfs = SFS(lr, 
           k_features=13, 
           forward=True, 
           floating=False, 
           scoring='neg_mean_squared_error',
           cv=5)

feature_names = ('zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat','medv')
sfs = sfs.fit(x_train, y_train, custom_feature_names = feature_names)


# In[34]:


pd.DataFrame.from_dict(sfs.get_metric_dict()).T.sort_values('avg_score',ascending=False)


# In[35]:


x_train_selected = x_train.iloc[:,[0, 1, 3, 6, 7, 9, 12]]
x_test_selected = x_test.iloc[:,[0, 1, 3, 6, 7, 9, 12]]
lr.fit(x_train_selected,y_train)

pred = lr.predict(x_test_selected)
mse=mean_squared_error(y_test,pred)
print('MSE: ', format(mse,'.2f'))
rsq = r2_score(y_test,pred)
print('R-square: ',format(rsq,'.4f'))


# # Backward Selection

# In[36]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

np.random.seed(0)
boston_data=pd.read_csv("Boston_Data.csv")

X = boston_data.drop('crim', axis = 1 )
Y = boston_data['crim']

#split dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=5)


# In[37]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
lr = LinearRegression()

sfs = SFS(lr, 
           k_features=1, 
           forward=False, 
           floating=False, 
           scoring='neg_mean_squared_error',
           cv=5)

feature_names = ('zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat','medv')
sfs = sfs.fit(x_train, y_train, custom_feature_names = feature_names)


# In[38]:


pd.DataFrame.from_dict(sfs.get_metric_dict()).T.sort_values('avg_score',ascending=False)


# In[39]:


x_train_selected = x_train.iloc[:,[0, 1, 3, 6, 7, 9, 12]]
x_test_selected = x_test.iloc[:,[0, 1, 3, 6, 7, 9, 12]]
lr.fit(x_train_selected,y_train)

pred = lr.predict(x_test_selected)
mse=mean_squared_error(y_test,pred)
rsq = r2_score(y_test,pred)
print('MSE: ', format(mse,'.2f'))
print('R-square: ',format(rsq,'.4f'))


# # Ridge Regression

# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error

np.random.seed(0)
boston_data=pd.read_csv("Boston_Data.csv")

X = boston_data.drop('crim', axis = 1 )
Y = boston_data['crim']


# In[41]:


#The Ridge() function has an alpha argument ( λ , but with a different name!) that is used to tune the model. 
#We'll generate an array of alpha values ranging from very big to very small, essentially covering the full range of scenarios 
#from the null model containing only the intercept, to the least squares fit
alphas = 10**np.linspace(5,-2,100)*0.5
alphas


# In[42]:


from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
Xscaled = scaler.fit_transform(X)


# In[43]:


ridge = Ridge()
coefs = []

for a in alphas:
    ridge.set_params(alpha = a)
    ridge.fit(Xscaled, Y)
    coefs.append(ridge.coef_)
np.shape(coefs)


# In[44]:


ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


# In[45]:


# Split data into training and test sets
x_train, x_test , y_train, y_test = train_test_split(Xscaled, 
                                                     Y, 
                                                     test_size=0.2, 
                                                     random_state=5)


# In[46]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
lambdas = np.linspace(0.01,100,num=1000)
scoresCV = []
for l in lambdas:
    RidgeReg = Ridge(alpha=l)
    RidgeReg.fit(x_train, y_train)    
    scoreCV = cross_val_score(RidgeReg, x_train, y_train, scoring='neg_mean_squared_error',
                             cv=KFold(n_splits=10, shuffle=True,
                                            random_state=1))
    scoresCV.append([l,-1*np.mean(scoreCV)])
df = pd.DataFrame(scoresCV,columns=['Lambda','Validation Error'])
df


# In[47]:


plt.plot(df.Lambda,df['Validation Error'])


# In[48]:


ridgecv = RidgeCV(alphas = alphas,cv = 10, scoring = 'neg_mean_squared_error')
ridgecv.fit(x_train, y_train)
print('Best alpha:', format(ridgecv.alpha_,'.4f'))


# In[49]:


ridgen = Ridge(alpha = ridgecv.alpha_)
ridgen.fit(x_train, y_train)

print('Mean Squared Error: ',format(mean_squared_error(y_test, ridgen.predict(x_test)),'.4f'))


# In[50]:


print('R-square: ',format(ridgecv.score(x_test, y_test),'.4f'))


# In[51]:


ridgen.fit(X, Y)
pd.Series(ridgen.coef_, index = X.columns)


# # Lasso Regression

# In[52]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error

boston_data=pd.read_csv("Boston_Data.csv")

X = boston_data.drop('crim', axis = 1 )
Y = boston_data['crim']

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
Xscaled = scaler.fit_transform(X)

# Split data into training and test sets
x_train, x_test , y_train, y_test = train_test_split(Xscaled, 
                                                     Y, 
                                                     test_size=0.2, 
                                                     random_state=5)


# In[53]:


lasso = Lasso(max_iter = 10000)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(x_train), y_train)
    coefs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


# In[54]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
lambdas = np.linspace(0.01,100,num=1000)
scoresCV = []
for l in lambdas:
    lassoReg = Lasso(alpha=l,max_iter=10000)
    lassoReg.fit(x_train, y_train)    
    scoreCV = cross_val_score(lassoReg, x_train, y_train, scoring='neg_mean_squared_error',
                             cv=KFold(n_splits=10, shuffle=True,
                                            random_state=1))
    scoresCV.append([l,-1*np.mean(scoreCV)])
df = pd.DataFrame(scoresCV,columns=['Lambda','Validation Error'])
df


# In[55]:


plt.plot(df.Lambda,df['Validation Error'])


# In[56]:


lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000)
lassocv.fit(x_train, y_train)
lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(x_train, y_train)
lassocv.alpha_


# In[57]:


print('Mean Squared Error: ',format(mean_squared_error(y_test, lasso.predict(x_test)),".4f"))


# In[58]:


print('R-square: ',format(lassocv.score(x_test,y_test),".4f"))


# In[59]:


pd.Series(lasso.coef_, index=X.columns)


# # KNN

# In[60]:


get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error

boston_data=pd.read_csv("Boston_Data.csv")

X = boston_data.drop('crim', axis = 1 )
Y = boston_data['crim']

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
Xscaled = scaler.fit_transform(X)

# Split data into training and test sets
x_train, x_test , y_train, y_test = train_test_split(Xscaled, Y, test_size=0.2, random_state=1)


# In[61]:


len(x_train)


# In[62]:


#for to find best k
a_=[]
for a in range(1,405):
    a_.append(a)
df=pd.DataFrame()
scr=[]
df["k"]=a_
for k in a_:
    knn = KNeighborsRegressor(n_neighbors = k)
    knn.fit(x_train,y_train)
    scr_=knn.score(x_test, y_test)
    scr.append(scr_)
df["score"]=scr
df.sort_values(by="score",ascending=False)


# In[63]:


knn = KNeighborsRegressor(n_neighbors = 22)
knn.fit(x_train,y_train)
print('Mean Squared Error: ',format(mean_squared_error(knn.predict(x_test),y_test),".4f"))


# In[64]:


print('R-square: ',format(knn.score(x_test, y_test),".4f"))


# In[65]:


fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(df['k'].values,df['score'])
ax.set_xlabel('NeighbourSize')
ax.set_ylabel('Score')
ax.tick_params(axis='x', labelsize=10)


# In[ ]:




