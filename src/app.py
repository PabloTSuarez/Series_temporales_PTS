#from utils import db_connect
#engine = db_connect()

# your code here
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error,r2_score
from pickle import dump


df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/alternative-time-series-project/main/sales.csv')
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.date
df = df.set_index('date')
df

plt.subplots(figsize=(10,5))
sns.lineplot(data=df)
plt.tight_layout()
plt.show()

decomposicion = seasonal_decompose(df,period=30)
trend = decomposicion.trend

plt.subplots(figsize=(10,5))
sns.lineplot(data=df)
sns.lineplot(data=trend)
plt.tight_layout()
plt.show()

estacional = decomposicion.seasonal

plt.subplots(figsize=(10,5))
sns.lineplot(data=df)
sns.lineplot(data=estacional)
plt.tight_layout()
plt.show()

def test_stationarity(timeseries):
    print("Resultados de la prueba de Dickey-Fuller:")
    dftest = adfuller(timeseries, autolag = "AIC")
    dfoutput = pd.Series(dftest[0:4], index = ["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
    for key,value in dftest[4].items():
        dfoutput["Critical Value (%s)"%key] = value
    return dfoutput

test_stationarity(df)
residuos = decomposicion.resid

plt.subplots(figsize=(10,5))
sns.lineplot(data=df)
sns.lineplot(data=residuos)
plt.tight_layout()
plt.show()


plot_acf(df)
plt.tight_layout()
plt.show()

df_stationary = df.diff().dropna()
test_stationarity(df_stationary)

model = auto_arima(df_stationary,seasonal=True,trace=True,m=7)
model.summary()
train_df = df_stationary[(df_stationary.index>pd.to_datetime('2023-07-25').date()) &(df_stationary.index<pd.to_datetime('2023-08-25').date())]
test_df = df_stationary[df_stationary.index>=pd.to_datetime('2023-08-25').date()]
test_df
model = auto_arima(train_df,seasonal=True,trace=True,m=7)
forecast = model.predict(10)

plt.subplots(figsize=(10,5))
sns.lineplot(data=test_df)
sns.lineplot(data=forecast)
plt.tight_layout()
plt.show()


mean_squared_error(test_df,forecast)
r2_score(test_df,forecast)

train_df = df[df.index<pd.to_datetime('2023-08-25').date()]
test_df = df[df.index>=pd.to_datetime('2023-08-25').date()]
model = auto_arima(train_df,seasonal=True,trace=True,m=7)
forecast = model.predict(10)

plt.subplots(figsize=(10,5))
sns.lineplot(data=test_df)
sns.lineplot(data=forecast)
plt.tight_layout()
plt.show()

mean_squared_error(test_df,forecast)
r2_score(test_df,forecast)

dump(model,open('../models/model_serie_temporal1.model','wb'))