import pandas as pd
from pandas import Series
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6

#converting data into timeseries
dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
data = pd.read_csv('FACEBOOK - NASDAQ.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)

#taking only the closing values
ts = data['Close']
ts = ts['2013-12-31':]
plt.plot(ts)
plt.title('Original TS Plot')
plt.show()

#penalising the higher values
ts_log = np.log(ts)

#exponential weighted mean average - quaterly
expweighted_avg = ts_log.ewm(halflife=30,min_periods=0,adjust=True,ignore_na=False).mean()
plt.plot(ts_log)
plt.plot(expweighted_avg, color='red')
plt.title('EWMA of TS')
plt.show()

#eliminating trends and seasonality
ts_log_ewma_diff = ts_log - expweighted_avg
ts_log_ewma_diff.dropna(inplace=True)
plt.plot(ts_log_ewma_diff)
plt.title('Stationary TS')
plt.show()

from pandas import Series
import datetime
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    
    #rolling statistics
    rolmean = timeseries.rolling(window=30, center=False).mean()
    rolstd = timeseries.rolling(window=30, center=False).std()

    #plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    #dickey-fuller test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value
    print(dfoutput)

test_stationarity(ts_log_ewma_diff)

from statsmodels.tsa.stattools import acf, pacf

#plot to determine q value
lag_acf = acf(ts_log_ewma_diff, nlags=30)
#pacf plot to determine p value
lag_pacf = pacf(ts_log_ewma_diff, nlags=30, method='ols')

plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_ewma_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_ewma_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.show()

plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_ewma_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_ewma_diff)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.show()

from statsmodels.tsa.arima_model import ARIMA

#AR model
model = ARIMA(ts_log_ewma_diff, order=(23, 0, 0))
results_AR = model.fit(disp=-1)
plt.plot(ts_log_ewma_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('AR Model fitted to Stationary TS')
plt.show()

predictions_AR_ewma_diff = pd.Series(results_AR.fittedvalues, copy=True)

#adding seasonality and trend back
predictions_AR_log = predictions_AR_ewma_diff + expweighted_avg

#getting back original values - removing log
predictions_AR = np.exp(predictions_AR_log)

predictions_AR_next_log = model.predict(results_AR.params, start=716, end=730)
predictions_AR_next_log = predictions_AR_next_log[1:]

for i, x in enumerate(predictions_AR_next_log):
    predictions_AR_next_log[i] = x + expweighted_avg[-1]

predictions_AR_next = np.exp(predictions_AR_next_log)
    
plt.plot(predictions_AR_next)
plt.title('Future Predictions')
plt.show()

df = predictions_AR.to_frame()
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in predictions_AR_next:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [i]

plt.plot(ts)
plt.plot(df)
plt.title('Complete predictions')
plt.show()
