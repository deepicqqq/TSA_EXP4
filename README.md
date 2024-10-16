# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
import pandas as pd\
import matplotlib.pyplot as plt\
from statsmodels.tsa.arima.model import ARIMA\
from sklearn.metrics import mean_squared_error

#Load the dataset\
data = pd.read_csv('DailyDelhiClimateTest.csv', parse_dates=['date'], index_col='date')

#Select the mean temperature time series\
temp_series = data['meantemp']

#Split the data into train and test sets\
train_size = int(len(temp_series) * 0.8)  # 80% for training, 20% for testing\
train, test = temp_series[:train_size], temp_series[train_size:]

#Fit the ARMA model (Auto Regressive Moving Average)\
#p: order of AR component (autoregression lag)\
#q: order of MA component (moving average lag)\
#Since ARMA is a special case of ARIMA (without differencing), we set d=0\
arma_model = ARIMA(train, order=(2, 0, 2)).fit()  # AR(2), MA(2), differencing (d=0)

#Make predictions for the test data\
predictions = arma_model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

#Calculate Mean Squared Error (MSE) for model evaluation\
mse = mean_squared_error(test, predictions)\
print(f'Mean Squared Error: {mse:.2f}')

#Plot the observed data and predictions\
plt.figure(figsize=(10, 6))\
plt.plot(test.index, test, label='Observed')\
plt.plot(test.index, predictions, label='Predicted', color='red')\
plt.title('ARMA Model for Mean Temperature in Delhi')\
plt.xlabel('Date')\
plt.ylabel('Temperature (°C)')\
plt.legend()\
plt.show()

#Forecast future values (e.g., next 30 days)\
forecast = arma_model.predict(start=len(temp_series), end=len(temp_series)+30, dynamic=False)

#Plot the forecast\
plt.figure(figsize=(10, 6))\
plt.plot(temp_series, label='Observed')\
plt.plot(forecast.index, forecast, label='Forecast', color='green')\
plt.title('ARMA Model Forecast for Mean Temperature in Delhi')\
plt.xlabel('Date')\
plt.ylabel('Temperature (°C)')\
plt.legend()\
plt.show()

#Print the forecasted values\
print(forecast)

OUTPUT:
SIMULATED ARMA(1,1) PROCESS:

![Screenshot 2024-10-16 110105](https://github.com/user-attachments/assets/b9adb338-52c0-4ec8-bc6c-82190de061c7)



Partial Autocorrelation

![Screenshot 2024-10-16 110113](https://github.com/user-attachments/assets/f55fd22f-ba96-4911-b39e-979e93c3567e)

Autocorrelation

![Screenshot 2024-10-16 110119](https://github.com/user-attachments/assets/46700113-cad8-437c-9603-ebc337e2e6b7)


SIMULATED ARMA(2,2) PROCESS:

![Screenshot 2024-10-16 110132](https://github.com/user-attachments/assets/fec7d44c-9c93-43c3-ae7f-d24dc0f0417d)

Partial Autocorrelation
![Screenshot 2024-10-16 110052](https://github.com/user-attachments/assets/1b62b807-4c0c-4348-86e6-16032e1f0c0a)

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
