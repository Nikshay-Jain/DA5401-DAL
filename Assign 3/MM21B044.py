# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error as mse


# In[2]:


df = pd.read_csv('Assignment3.csv')
for col in df.columns:
    df[col] = df[col].astype(float)
df


# In[3]:


def SSE(y, yhat):
    return np.sum((y-yhat)**2)


# ### Task 1

# In[4]:


xx = np.array(df[['x1','x2','x3','x4','x5']])
yy = np.expand_dims(df['y'], 1)


# In[5]:


model1 = LinearRegression()
model1.fit(xx, yy)
yhat = model1.predict(xx)
loss = mse(yy, yhat)**0.5
print("Beta :", model1.coef_, " Bias :", model1.intercept_)
print("RMSE :",loss)


# ### Task 2

# In[6]:


df.describe()


# #### Check correlation matrix

# In[7]:


all_data = np.concatenate((xx, yy), axis=1)
corr = np.corrcoef(all_data.T)
print(corr)


# ### Visualize the relationships through a pair-plot.

# In[8]:


sb.pairplot(df)


# It is evident that x1 and x4 are highly correlated with y and with each other while x5 is proportional to x2^2 with x2 being slightly correlated with y so there can be a potential correlation in x5^0.5 and y. x3 seems to have no correlation with y so better to drop it. Since x4 is slightly more correlated with y than x1, we can choose to drop x1 as well.

# In[9]:


x2 = np.array(df['x2']).reshape(-1,1)
x4 = np.array(df['x4']).reshape(-1,1)
x5 = np.array(df['x5']).reshape(-1,1)
y = np.array(df['y']).reshape(-1,1)


# #### Using only x4 to predict y

# In[10]:


model21 = LinearRegression()
model21.fit(x4, y)
yhat21 = model21.predict(x4)
loss21 = mse(y, yhat21)**0.5

print("Beta :", model21.coef_, " Bias :", model21.intercept_)
print("RMSE :",loss21)


# In[11]:


plt.plot(x4,yhat21,'r')
plt.scatter(x4,y)
plt.xlabel('x4')
plt.ylabel('y')
plt.title('y vs x4')
plt.show()


# The loss is pretty high here!
# 
# Let's go for feature engineering now: form a feature $x_6 = \alpha_1 x_5^{1/2} + \alpha_2$ or $x_6 = \alpha_1 x_5^{1/4} + \alpha_2$

# In[12]:


x6 = np.array(x5**0.5)

plt.scatter(x2, x6)
plt.xlabel('x2')
plt.ylabel('$x_5^{1/2}$')
plt.title('Mapping x5 to x2')
plt.show()


# The modulus while taking even root is creating an issue - fix it

# In[13]:


x6 = [x6[i] * -1 if x2[i] < 0 else x6[i] for i in range(len(x2))]

plt.scatter(x2, x6)
plt.xlabel('x2')
plt.ylabel('$x_5^{1/2}$')
plt.title('Mapping x5 to x2 without mod as x6')
plt.show()


# The fit is sufficiently linear so better we go with sq root function

# In[14]:


x6 = np.array(x6).flatten()
df1 = df.assign(x6=x6)


# Linear search for optimum power of x4 as it shows quite high correlation with y

# In[15]:


alpha_range = np.arange(0, 6)
alphas_mse = np.empty((len(alpha_range),2))
best_alpha = None
lowest_mse = float('inf')

for i, alpha in enumerate(alpha_range):
    df1['x7'] = df1['x1']*df1['x4']
    df1['x8'] = df1['x4']**alpha

    xx_new = np.array(df1[['x1','x2','x3','x4','x5','x6','x7','x8']])
    yy_new = np.expand_dims(df1['y'], 1)

    X_train, X_test, y_train, y_test = train_test_split(xx_new, yy_new, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse_loss = mse(y_test, y_pred)

    alphas_mse[i] = [alpha, mse_loss]
    if mse_loss < lowest_mse:
        best_alpha = alpha
        lowest_mse = mse_loss

print(f"Best alpha: {best_alpha}")
print(f"Lowest MSE: {lowest_mse**0.5}")


# In[16]:


plt.plot(alphas_mse[:,0],alphas_mse[:,1])
plt.xlabel('Powers of x4')
plt.ylabel('MSE of testing data')
plt.title('Linear search for powers of x4')
plt.show()


# In[17]:


df1 = df1[['x1','x2','x3','x4','x5','x6','x7','x8','y']]
sb.pairplot(df1)


# In[18]:


xx_new = np.array(df1[['x1','x2','x3','x4','x5','x6','x7','x8']])
yy_new = np.expand_dims(df1['y'], 1)

X_train, X_test, y_train, y_test = train_test_split(xx_new, yy_new, test_size=0.2, random_state=42)


# In[19]:


model2 = LinearRegression()
model2.fit(X_train, y_train)
yhat_train = model2.predict(X_train)
rmse_train = mse(y_train, yhat_train)**0.5

print("Beta :", model2.coef_, " Bias :", model2.intercept_)


# In[20]:


yhat_test = model2.predict(X_test)
rmse_test = mse(y_test,yhat_test)**0.5
r2 = r2_score(y_test, yhat_test)

print("R2 score:",r2)
print("RMSE on training data:",rmse_train)
print('RMSE on testing data:',rmse_test)
print("SSE on training data:",SSE(y_train, yhat_train))
print("SSE on testing data:",SSE(y_test, yhat_test))


# In[21]:


n = len(y_test)
p = X_test.shape[1] 
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
print("Adjusted R2 score:",adjusted_r2)


# ### Task 4

# Running LazyRegressor on original data

# In[22]:


X = df[['x1', 'x2', 'x3','x4','x5']]
y = df['y']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


from lazypredict.Supervised import LazyRegressor

reg_orig = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg_orig.fit(X_train1, X_test1, y_train1, y_test1)
best_model = models.sort_values(by='R-Squared', ascending=False).iloc[0]

print("Lazy regressor on original data:")
print(models)
print("Best Model:", best_model)


# The given data suggests that the least RMSE reached is 43.02 by LASSO model while LinearRegression gives an RMSE of 44.73. The difference is mainly by the weights made 0 for useless features by LASSO by the application of L1 regularisation. Lets try once by the new dataset we prepared.

# Running LazyRegressor on new data after adding the new features

# In[24]:


reg_new = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models1, predictions1 = reg_new.fit(X_train, X_test, y_train, y_test)
best_model1 = models1.sort_values(by='R-Squared', ascending=False).iloc[0]

print("Lazy regressor after adding neew features in data:")
print(models1)
print("Best Model:", best_model1)


# The LASSOLarsCV and LASSOLarsIC models had the lowest RMSE of 0.84, which matched the LinearRegression model. The regularization of LASSO models lowers overfitting and enhances generalization, resulting in their effectiveness while having equal RMSE values. This regularization aids in managing model complexity and increasing robustness.

# In[ ]:




