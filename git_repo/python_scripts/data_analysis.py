import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix

import setup_env
from setup_env import *

engine = setup_env.get_database()
print(engine)
try:
    con = engine.raw_connection()
    con.cursor().execute("SET SCHEMA '{}'".format("here_traffic"))
except:
    print("Con: DB Verbindung prüfen!") 
    pass

sql_query ="""
    SELECT link_id, confidence
    FROM here_traffic.car_60minuten ta
    --WHERE link_id = '53226296'
    --AND dir_travel != 'B'
"""

#53226296
pd_read = pd.read_sql_query(sql_query, con)

df = pd.DataFrame(
    pd_read,
    columns=[
        "link_id",  
        "confidence",
    ],
)

df["rank"] = (df["confidence"].rank(ascending=True))
print("\ndf shape, ndim:\n",df.shape, df.ndim)
print("Dataframe:\n",df,"\nMedian:",df["confidence"].median(),"\nMean:",df["confidence"].mean())



sns.set()
ax = sns.lineplot(x="rank",y="confidence", data=df)
ax.set(xlabel='Link', ylabel='Confidence')

plt.show()