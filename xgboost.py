#!/usr/bin/env python
# coding: utf-8

# # Hourly Time Series Forecasting using XGBoost
# 
# [If you haven't already first check out my previous notebook forecasting on the same data using Prophet](https://www.kaggle.com/robikscube/hourly-time-series-forecasting-with-prophet)
# 
# In this notebook we will walk through time series forecasting using XGBoost. The data we will be using is hourly energy consumption.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')


# In[2]:


from get_holidays import get_holidays


# In[3]:


import datetime


# In[4]:


from sys import exit


# In[5]:


import setup_env


# In[6]:


from math import isnan


# In[7]:


rows = "*"
sel_link_id = 841461124
sel_min_confidence = 0
sel_max_weekday = 8
sel_func_classes = ('4','3')
sel_dir_travel = 'F'


# In[8]:


sql_query = f"""
    SELECT tr.*  
    FROM here_traffic.stuttgart_traffic tr
    JOIN here_streets.fc_streets_all_2018q3 st on tr.link_id = st.link_id
    WHERE tr.link_id = {sel_link_id}
    AND tr.confidence > {sel_min_confidence}
    AND tr.weekday_n < {sel_max_weekday}
    --AND st.func_class in {sel_func_classes}
    AND tr.dir_travel = 'F'
    LIMIT 100000
"""


# In[9]:


engine = setup_env.get_database()
print(engine)


# In[10]:


try:
    con = engine.raw_connection()
    con.cursor().execute("SET SCHEMA '{}'".format("here_traffic"))
except:
    print("Error: DB Verbindung prüfen!") 
    exit


# In[11]:


pd_read = pd.read_sql_query(sql_query, con)


# In[12]:


df = pd.DataFrame(
    pd_read,
    columns=[
        "id_pk",
        "link_id",
        "dir_travel",
        "mean_kmh",     #2
        "datum_zeit",   #5
        "weekday_n",    #6
        "epoch_60",     #7
        "confidence",
        "count_n",
    ],
)


# In[ ]:


df


# In[13]:


if df.shape[0] < 1: print('Failure loading data'),exit()


# Drop duplicate entries if existing.

# In[14]:


df.drop_duplicates(inplace=True)


# Pandas *should* infer correct data types, but can mistake numericals for str.  
# Hardcode Pandas data types. 

# In[15]:


df.astype({'id_pk':'int64',
            'mean_kmh':'int8',
            'weekday_n':'int8',
            'epoch_60':'int8',
            'count_n':'int32',          
          }).dtypes


# ---
# The input data is **not** gap-filled yet. 
# Missing dates/values will be interpolated.  
# To prevent excessive gap-filling (i.e. most of the data being interpolated),  
# links with less than 75% existing data will no be calculated

# In[16]:


share_data = (df.shape[0]/(365*24)*100)

if share_data <= 75:
    exit(f"Share of existing data is only {share_data}%. Exiting.")
else: 
    print(f"Share of existing data for this link: {share_data}%")


# ---  
# #### Check speed data for sanity 
# Mean vehicle speed data can be errorenous (i.e. in excess of 150 kph within town center).  
# Calculate the (upper) 99th percentile and clip values above it.

# In[17]:


quantile_99 = df["mean_kmh"].quantile(0.99)
df = df[df["mean_kmh"] < quantile_99]
print(f"Discarding values above {quantile_99} kph")


# Set the datetime column as index and sort the dataframe by it.

# In[ ]:


df.set_index('datum_zeit', inplace=True, drop=True)
df.sort_index(inplace=True)


# Create a range of date(-times) over the entire time frame of interest (2018-05-01 00:00:00 to 2019-04-30 23:00:00).  
# 
# Using the datetime range with Pandas *reindex* on the Dataframe fills in potentially missing dates / rows. 

# In[ ]:


fill_index = pd.date_range('2018-05-01 00:00:00', '2019-04-30 23:00:00', freq='1H')

df = df.reindex(fill_index)


# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


df.drop_duplicates(inplace=True)





df['link_id']=sel_link_id
df['dir_travel']=df['dir_travel'].fillna(method='backfill')
df['epoch_60']=df.index.hour
df['weekday_n']=df.index.dayofweek+1
df["hourweek"] = (df["weekday_n"]-1)*24+df["epoch_60"]
df["hourweek"] = pd.to_numeric(df.hourweek, errors='coerce')
df['confidence'] = df['confidence'].fillna(5)

df['count_diff'] = df['count_n'].diff()
df['count_diff'] = df['count_diff'].fillna(0)

count_mean = df['count_n'].groupby(df['hourweek']).mean()

for index, row in df.iterrows():
    m_idx = row['hourweek']
    if isnan(row['count_n']):
        row['count_n'] = count_mean[m_idx]
        df['count_n'].at[index] = count_mean[m_idx].round()
        
remain_nan = df['count_n'].isna().sum()

if remain_nan > 0:
    print("%i Remaining vehicle count cols without value. Aborting." % (remain_nan))
    exit(1)   


# In[ ]:





# In[88]:





# In[19]:


df.shape[0]


# In[20]:


count_mean = df['count_n'].groupby(df['hourweek']).mean()


# In[ ]:





# In[21]:


for index, row in df.iterrows():
    m_idx = row['hourweek']
    if isnan(row['mean_kmh']):
        row['mean_kmh'] = count_mean[m_idx]
        df['mean_kmh'].at[index] = count_mean[m_idx].round(1)


# In[22]:


color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
_ = df['mean_kmh'].plot(style='.', figsize=(40,5), color=color_pal[0], title='Durchschnittsgeschwindigkeit Neckartor')


# In[23]:


df.loc[:,'date'] = df.index.date


# In[24]:


df.loc[:,'holiday'] = np.nan


# In[25]:


holidays = list(get_holidays(2018,2019))


# In[26]:


for i in range(-1,3): 
    df.loc[df['date'].isin([x+ datetime.timedelta(days=i) for x in holidays]), ['holiday']] = i


# Create time-sshifted rows for mean traffic speed from now until 72 hours in the past.
# Drop last 72 hours of dataframe.

# In[ ]:


for i in range(1,73):
    df[f'speed_shift_{i}'] = df['mean_kmh'].shift(i)
    df[f'weekday_shift_{i}'] = df['weekday_n'].shift(i)
    df[f'epoch_60_]
df.drop(df.tail(i).index,inplace=True)


# In[106]:





# In[105]:


df.columns


# In[134]:


df = df[['mean_kmh', 'weekday_n', 'epoch_60','confidence','holiday','hourweek']]


# In[135]:


df


# In[136]:


for i in df.columns:
    for j in range(1,73):
        df[f'{i}_shift_{j}'] = df[f'{i}'].shift(j)    


# In[137]:


df[['mean_kmh','mean_kmh_shift_1','mean_kmh_shift_2','mean_kmh_shift_3']]


# In[138]:


df = df.infer_objects()


# In[139]:


split_date = '2019-03-20 23:00:00'


# X_train = mean_kmh, weekday, epoch, hourweek, holiday  
# Y_train = mean_kmh
# 

# In[140]:


#X_train = df[['mean_kmh', 'weekday_n', 'epoch_60', 'hourweek', 'holiday']].loc[:split_date]


# In[141]:


X_train = df.loc[:split_date]
del X_train['mean_kmh']


# In[142]:


y_train = df['mean_kmh'].loc[:split_date]


# In[150]:


X_test = df.loc[split_date:]
del X_test['mean_kmh']


# In[151]:


y_test = df['mean_kmh'].loc[split_date:]


# # Train/Test Split
# Cut off the data after 2015 to use as our validation set.

# In[152]:


X_train


# In[153]:


y_train


# In[154]:


# split_date = '01-Jan-2015'
# pjme_train = pjme.loc[pjme.index <= split_date].copy()
# pjme_test = pjme.loc[pjme.index > split_date].copy()


# In[155]:


# _ = pjme_test \
#     .rename(columns={'PJME_MW': 'TEST SET'}) \
#     .join(pjme_train.rename(columns={'PJME_MW': 'TRAINING SET'}), how='outer') \
#     .plot(figsize=(15,5), title='PJM East', style='.')


# In[ ]:





# # Create Time Series Features

# In[156]:


# def create_features(df, label=None):
#     """
#     Creates time series features from datetime index
#     """
#     df['date'] = df.index
#     df['hour'] = df['date'].dt.hour
#     df['dayofweek'] = df['date'].dt.dayofweek
#     df['quarter'] = df['date'].dt.quarter
#     df['month'] = df['date'].dt.month
#     df['year'] = df['date'].dt.year
#     df['dayofyear'] = df['date'].dt.dayofyear
#     df['dayofmonth'] = df['date'].dt.day
#     df['weekofyear'] = df['date'].dt.weekofyear
    
#     X = df[['hour','dayofweek','quarter','month','year',
#            'dayofyear','dayofmonth','weekofyear']]
#     if label:
#         y = df[label]
#         return X, y
#     return X


# In[157]:


# X_train, y_train = create_features(pjme_train, label='PJME_MW')
# X_test, y_test = create_features(pjme_test, label='PJME_MW')


# In[ ]:





# In[158]:


print("Training input columns,rows: ",X_train.shape[1],y_train.shape[0])
print("Training target rows: ",y_train.shape[0])


# # Create XGBoost Model

# In[160]:


reg = xgb.XGBRegressor(n_estimators=5000,
                        n_jobs = 10,
                        gamma = 1e-5,)

reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=100,
       verbose=True) # Change verbose to True if you want to see it train


# ## Feature Importances
# Feature importance is a great way to get a general idea about which features the model is relying on most to make the prediction. This is a metric that simply sums up how many times each feature is split on.
# 
# We can see that the day of year was most commonly used to split trees, while hour and year came in next. Quarter has low importance due to the fact that it could be created by different dayofyear splits.

# In[163]:


_ = plot_importance(reg, height=0.9, max_num_features=20)


# In[164]:


df_test = df[split_date:]


# In[165]:


df_test['mean_kmh_predict'] = reg.predict(X_test)


# In[166]:


df_test['mean_kmh_predict']


# In[167]:


df_all = pd.concat([df, df_test], sort=False)


# In[168]:


df_all[['mean_kmh','mean_kmh_predict']].iloc[-20:]


# In[175]:


plt.figure(figsize=(40, 4))
plt.plot(df_all['mean_kmh'].iloc[-480:], label='Mean speed: Truth')
plt.plot(df_all['mean_kmh_predict'].iloc[-480:], label='Mean speed: Predicted',linestyle='dashed')
plt.title("XGBoost Prediction")
plt.legend()
plt.show()


# In[59]:


df_all['mean_kmh_predict'].iloc[-724:]


# In[47]:


mean_squared_error(y_true=df_all['mean_kmh']['2019-04-01':],
                   y_pred=df_all['mean_kmh_predict']['2019-04-01':])


# In[ ]:


df_all['mean_kmh_predict']['2019-04-30':]


# # Look at first month of predictions

# In[ ]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,
                                              style=['-','.'])
ax.set_xbound(lower='01-01-2015', upper='02-01-2015')
ax.set_ylim(0, 60000)
plot = plt.suptitle('January 2015 Forecast vs Actuals')


# 

# In[ ]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,
                                              style=['-','.'])
ax.set_xbound(lower='01-01-2015', upper='01-08-2015')
ax.set_ylim(0, 60000)
plot = plt.suptitle('First Week of January Forecast vs Actuals')


# In[ ]:


f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,
                                              style=['-','.'])
ax.set_ylim(0, 60000)
ax.set_xbound(lower='07-01-2015', upper='07-08-2015')
plot = plt.suptitle('First Week of July Forecast vs Actuals')


# # Error Metrics On Test Set
# Our RMSE error is 13780445  
# Our MAE error is 2848.89  
# Our MAPE error is 8.9%

# In[ ]:


mean_squared_error(y_true=pjme_test['PJME_MW'],
                   y_pred=pjme_test['MW_Prediction'])


# In[ ]:


mean_absolute_error(y_true=pjme_test['PJME_MW'],
                   y_pred=pjme_test['MW_Prediction'])


# I like using mean absolute percent error because it gives an easy to interperate percentage showing how off the predictions are.
# MAPE isn't included in sklearn so we need to use a custom function.

# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


mean_absolute_percentage_error(y_true=pjme_test['PJME_MW'],
                   y_pred=pjme_test['MW_Prediction'])


# # Look at Worst and Best Predicted Days

# In[ ]:


pjme_test['error'] = pjme_test['PJME_MW'] - pjme_test['MW_Prediction']
pjme_test['abs_error'] = pjme_test['error'].apply(np.abs)
error_by_day = pjme_test.groupby(['year','month','dayofmonth'])     .mean()[['PJME_MW','MW_Prediction','error','abs_error']]


# In[ ]:


# Over forecasted days
error_by_day.sort_values('error', ascending=True).head(10)


# Notice anything about the over forecasted days? 
# - #1 worst day - July 4th, 2016 - is a holiday. 
# - #3 worst day - December 25, 2015 - Christmas
# - #5 worst day - July 4th, 2016 - is a holiday.   
# Looks like our model may benefit from adding a holiday indicator.

# In[ ]:


# Worst absolute predicted days
error_by_day.sort_values('abs_error', ascending=False).head(10)


# The best predicted days seem to be a lot of october (not many holidays and mild weather) Also early may

# In[ ]:


# Best predicted days
error_by_day.sort_values('abs_error', ascending=True).head(10)


# # Plotting some best/worst predicted days

# In[ ]:


f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(10)
_ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,
                                              style=['-','.'])
ax.set_ylim(0, 60000)
ax.set_xbound(lower='08-13-2016', upper='08-14-2016')
plot = plt.suptitle('Aug 13, 2016 - Worst Predicted Day')


# This one is pretty impressive. SPOT ON!

# In[ ]:


f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(10)
_ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,
                                              style=['-','.'])
ax.set_ylim(0, 60000)
ax.set_xbound(lower='10-03-2016', upper='10-04-2016')
plot = plt.suptitle('Oct 3, 2016 - Best Predicted Day')


# In[ ]:


f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(10)
_ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,
                                              style=['-','.'])
ax.set_ylim(0, 60000)
ax.set_xbound(lower='08-13-2016', upper='08-14-2016')
plot = plt.suptitle('Aug 13, 2016 - Worst Predicted Day')


# # Up next?
# - Add Lag variables
# - Add holiday indicators.
# - Add weather data source.

# In[ ]:




