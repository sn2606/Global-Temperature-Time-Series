#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


gt = pd.read_csv('GlobalTemperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
gt.dropna(inplace = True)
gt.head()


# In[3]:


df = gt.reset_index(drop=True)


# In[4]:


df


# # VISUALIZATIONS

# In[5]:


col = [gt.columns[0], gt.columns[2], gt.columns[4], gt.columns[6]]
col


# In[6]:


fig = plt.figure(figsize = (20, 10))
axes = fig.add_axes([0, 0, 1, 1])
axes.plot(col[0], data = gt, color = 'y')
axes.plot(col[1], data = gt, color = 'r')
axes.plot(col[2], data = gt, color = 'b')
axes.plot(col[3], data = gt, color = 'c')
axes.set_title('Line Plot visualization of multivariate time series')
axes.set_xlabel('Row no')
axes.set_ylabel('Val')
axes.legend()
fig.savefig('Output/lineplot.png', bbox_inches = 'tight')


# In[7]:


gt[col].plot(subplots=True, figsize=(20, 10))
plt.savefig('Output/Linesubplots.png', bbox_inches = 'tight')


# In[8]:


fig = plt.figure(figsize = (20, 10))
axes = fig.add_axes([0, 0, 1, 1])
sns.distplot(gt[col[0]], ax = axes, color = 'y')
sns.distplot(gt[col[1]], ax = axes, color = 'r')
sns.distplot(gt[col[2]], ax = axes, color = 'b')
sns.distplot(gt[col[3]], ax = axes, color = 'g')
axes.legend(col)
fig.savefig('Output/distplot.png', bbox_inches = 'tight')


# In[9]:


colors = [['g', 'r'], ['b', 'k']]
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 10))
plt.tight_layout()
data = np.reshape(col, (2, 2))

for i in range(2):
    for j in range(2):
        sns.distplot(gt[data[i][j]], ax = axes[i][j], hist_kws=dict(edgecolor= 'k', linewidth=2), color = colors[i][j])
        
fig.savefig('Output/distsubplot.png', bbox_inches = 'tight')


# In[10]:


groups = gt[col[0]].groupby(pd.Grouper(freq='A'))


# In[11]:


LandAverageTemperature = pd.DataFrame()
for name, group in groups:
    LandAverageTemperature[name.year] = group.values
    
LandAverageTemperature


# In[12]:


LandAverageTemperature[LandAverageTemperature.columns[65:]].boxplot(figsize = (30, 20))
plt.xlabel('Years')
plt.title('Boxplot Visualization of Land Average temperatures 1915-2015')
plt.xticks(rotation = 90)
plt.savefig('Output/boxplot.png', bbox_inches = 'tight')


# In[13]:


groups = gt[col[3]].groupby(pd.Grouper(freq='A'))
LandAndOceanAverageTemperature = pd.DataFrame()
for name, group in groups:
    if(name.year > 2005):
        LandAndOceanAverageTemperature[name.year] = group.values

# LandAndOceanAverageTemperature.columns
LandAndOceanAverageTemperature = LandAndOceanAverageTemperature.transpose()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
LandAndOceanAverageTemperature.columns = months
LandAndOceanAverageTemperature


# In[14]:


LandAndOceanAverageTemperature.boxplot(figsize = (20, 10))
plt.xlabel('Months')
plt.ylabel('Readings')
plt.title('Box plot visualization of monthly global average temperature distribution 2006-15')
plt.savefig('Output/monthlyboxplot.png', bbox_inches = 'tight')


# In[15]:


groups = gt[col[2]].groupby(pd.Grouper(freq='A'))
LandMinTemperature = pd.DataFrame()
for name, group in groups:
    LandMinTemperature[name.year] = group.values
# years = years.T
plt.matshow(LandMinTemperature, interpolation=None, aspect='auto')
plt.xlabel('Dataframe heatmap visualization')
plt.savefig('Output/matviz.png', bbox_inches = 'tight')


# In[16]:


fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 10))
plt.tight_layout()
pd.plotting.lag_plot(gt[col[0]], ax = axes[0][0])
axes[0][0].set_title('LandAverageTemperature')
pd.plotting.lag_plot(gt[col[1]], ax = axes[0][1])
axes[0][1].set_title('LandMaxTemperature')
pd.plotting.lag_plot(gt[col[2]], ax = axes[1][0])
axes[1][0].set_title('LandMinTemperature')
pd.plotting.lag_plot(gt[col[3]], ax = axes[1][1])
axes[1][1].set_title('LandAndOceanAverageTemperature')
plt.savefig('Output/lagplot.png')


# In[17]:


fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
sns.heatmap(gt.corr(), annot=True)
fig.savefig('Output/correlation_heatmap.png', bbox_inches = 'tight')


# In[18]:


y = gt[col[0]]
y


# In[19]:


import statsmodels.api as sm
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
# tsa = time series analysis
fig = decomposition.plot()
fig.savefig('Output/decomposition.png')


# # CHECKING FOR STATIONARITY

# In[20]:


gt.info()


# In[21]:


df1, df2 = gt[0:996], gt[996:]
m1, m2 = df1.mean(), df2.mean()
v1, v2 = df1.var(), df2.var()
mv = pd.DataFrame([m1, m2, v1, v2])
mv = mv.T
mv.columns = ['m1', 'm2', 'v1', 'v2']
mv


# In[22]:


# Null Hypothesis (H0): If failed to be rejected, it suggests the time series has a unit root, 
#     meaning it is non-stationary. It has some time dependent structure.
# Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests the time series does not have a unit root, 
#     meaning it is stationary. It does not have time-dependent structure.
# p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
# p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

from statsmodels.tsa.stattools import adfuller
X = gt[col[0]].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
    
# time series is non stationary


# In[23]:


# Null Hypothesis: The process is trend stationary.
# Alternate Hypothesis: The series has a unit root (series is not stationary).
# Test for stationarity: If the test statistic is greater than the critical value, 
#     we reject the null hypothesis (series is not stationary). If the test statistic is 
#     less than the critical value, if fail to reject the null hypothesis (series is stationary). 


from statsmodels.tsa.stattools import kpss
result = kpss(X)
print('KPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[3].items():
    print('\t%s: %.3f' % (key, value))
    
# series is non-stationary


# # CONVERTING NON-STATIONARY TIME SERIES TO STATIONARY
# 
# ## LOG TRANSFORM AND DIFFERENCING

# In[24]:


gt1 = np.log(gt)
gt1[col].plot()


# # ARIMA

# In[25]:


import itertools

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[26]:


import warnings
warnings.filterwarnings('ignore')


# In[27]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[28]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[29]:


results.plot_diagnostics(figsize=(16, 8))
plt.savefig('Output/arima-stationtrue.png', bbox_inches = 'tight')


# In[30]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[31]:


results.plot_diagnostics(figsize=(16, 8))
plt.savefig('Output/arima-stationfalse.png', bbox_inches = 'tight')


# In[32]:


LandAverageTemperature[LandAverageTemperature.columns[150:]]


# In[33]:


pred = results.get_prediction(start=pd.to_datetime('2015-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2000':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Land Average Temperature')
plt.legend()

plt.savefig('Output/forecast.png')


# In[34]:


y_forecasted = pred.predicted_mean
y_truth = y['2015-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[ ]:




