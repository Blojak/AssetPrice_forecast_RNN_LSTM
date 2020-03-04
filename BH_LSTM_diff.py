# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:28:08 2020

@author: benja
"""

'''
Description: This program uses an artificial recurrent neural network
called "Long Short Term Memory" (LSTM) to predict the closing stock price
of corporation Apple using the past 60 day stock price
'''

# Import libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import metrics
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import matplotlib.dates as mdates
from keras.models import model_from_json # Import to save the results
import json
from keras.models import load_model


# Get the stock quote
# df = web.DataReader('AAPL', data_source='yahoo', start= '2012-01-01',
#                     end = '2019-12-17')

# =============================================================================
# Load Data from Yahoo (B-Share)
# =============================================================================
df = web.DataReader('BRK-B', data_source='yahoo', start= '2012-01-01',
                    end = '2020-02-26')

# =============================================================================
# Plot Time Series (Close Price)
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df.index, df['Close'],lw=1.5)
ax.set_xlabel('Date', fontsize =12)
ax.set_ylabel('Close Price USD ($)', fontsize = 12)
ax.set_title('Close Price History')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# ax.xaxis.set_major_locator(mdates.YearLocator())
# ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
plt.savefig('BH_close_price.eps')
ax.grid(True)
plt.show()


# =============================================================================
# Pre-Processing
# =============================================================================
# Create new dataframe just with the close data

data = df.filter(['Close'])
# # equivalent to 
# data1 = df[['Close']]

# Convert the dataframe to a numpy array
dataset = data.values


# import statsmodels.formula.api as smf            # statistics and econometrics
# import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

sm.graphics.tsa.plot_acf(dataset, lags=70)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.draw()
plt.savefig('ACF.png')
plt.tight_layout()
plt.show()


# fig, ax = plt.subplots()
sm.graphics.tsa.plot_pacf(dataset, lags=30)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('PACF.png')
plt.tight_layout()
plt.show()


'''
Augmented Dickey Fuller Test:
    - H0: states there is the presence of a unit root.
    - HA: states there is no unit root. In other words, Stationarity exists

Interpretation:
    the test statistic -0.9305 is greater than any critical values, therefore we 
    FAIL to reject H0 ---> there is a unit root 
    
    In addition, the p-value 0.778 states a high likelihood to reject the null
    even though a unit root exists
    
    ADF is left skewed, right steep
'''
# =============================================================================
# Integrate first-order 
# Here we use DataFrame-Shift ---> only in panda, not numpy, thus you need a
# shape of (x,) and not (x,1)
# =============================================================================

dataset_diff = data.Close.diff()

dataset_diff[0] = 0
'''
Notice, one can also use the numpy command .diff, e.g.
last_60_day_diff = np.diff(np.squeeze(last_60_day))
to get first differences

'''

dataset_diff = dataset_diff.values

Adickfuller_i1         = adfuller(dataset_diff)#, autolag = 'AIC'
ADF_stat_i1            = Adickfuller_i1[0]
ADF_p_value_i1         = Adickfuller_i1[1]
ADF_lags_used_i1       = Adickfuller_i1[2]
ADF_no_obs_i1          = Adickfuller_i1[3]
print('ADF Statistic: %f' % Adickfuller_i1[0])
print('p-value: %f' % Adickfuller_i1[1])
print('Critical Values:')
for key, value in Adickfuller_i1[4].items():
	print('\t%s: %.3f' % (key, value))




'''
Split into training and test/validation sets
This is also do-able with sklearn, check it out
'''


dataset_diff = dataset_diff.reshape(-1,1)


# Get the number of rows to train the model on
# Here we choose 80% training - 20% testing/validating
training_data_len = math.ceil(len(dataset_diff)*.8)


# Scale the data to a standard normal distribution
scaler = MinMaxScaler(feature_range = (0,1)) 
scaled_data = scaler.fit_transform(dataset_diff)

# Check standardized data
# fig, ax = plt.subplots()
# ax.hist(scaled_data)
# ax.set_ylabel('Close Price USD ($)', fontsize = 18)
# plt.show()
'''
- Create the training data set
- Create the scaled training data set
'''



train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

# memory effects, assume 60 for the first instance
ac = 60 

for i in range(ac, len(train_data)):
    x_train.append(train_data[i-ac:i, 0])
    '''
    Note: iterations:
        - 1.) from 0 to 60
        - 2.) from 1 to 61
        - 3.) from 2 to 62
    '''
    y_train.append(train_data[i,0])
    # len(y_train) = training_data_len - 60, since there is no history available
    # for the first 60 entries
    # Take the last 60 dates to forecast date 61

# Convert to numpy arrays, required for the LSTM model        
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape Data (required for LSTM)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
'''
The input to every LSTM layer must be three-dimensional.

The three dimensions of this input are:
    Samples. One sequence is one sample. A batch is comprised of one or more 
    samples.
    Time Steps. One time step is one point of observation in the sample.
    Features. One feature is one observation at a time step.

This means that the input layer expects a 3D numpy array of data when fitting the 
model and when making predictions, even if specific dimensions of the array 
contain a single value, e.g. one sample or one feature.

In our example: 
    - 1580 
    - 60 time steps
    - predict 1 target --> the close price
'''
# The ...,1 is since we have just one close price
# Built the LSTM model
model = Sequential()
# model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, activation = 'relu', return_sequences=True, input_shape=(x_train.shape[1],1)))

model.add(LSTM(50, return_sequences = False))

model.add(Dropout(0.5)) # [0.2,0.5]
# model.add(layers.Dropout(0.4)) # [0.2,0.5]


model.add(Dense(25))
model.add(Dense(1))


'''
Dense: implements the operation: output = activation(dot(input, kernel) + bias)
 where activation is the element-wise activation function passed as the 
 activation argument, kernel is a weights matrix created by the layer, and bias 
 is a bias vector created by the layer (only applicable if use_bias is True).

LSTM: Long Short-Term Memory Network
The model:
    Is an extension of RNN. An RNN recognozes clusters in time and is used when
    the variable/s is/are autocorrelated or when a logical cycle is observable,
    e.g. If the cllinet ate pizza yesterday, he/she will eat pasta today. The past
    has shown that when the client ate pasta, fries are following ....
    The LSTM is a RNN with memory effects. It memorizes the the cycles

50: The layer has 50 neurons

return_sequences=True: because we use further layers

return_sequences = Flase: no LSTM layer follows in the model architecture

verbose: {0,1,2} displys the progressing bar

# model.add(Dropout(0.5)): Buch, S.149 Dropout consists in randomly setting a fraction 
rate of input units to 0 at each update during training time, which helps 
prevent overfitting.

Every LSTM layer should be accompanied by a Dropout layer. This layer will help 
to prevent overfitting by ignoring randomly selected neurons during training, 
and hence reduces the sensitivity to the specific weights of individual neurons. 
20% is often used as a good compromise between retaining model accuracy and 
preventing overfitting.

'''

# =============================================================================
# Compile (configures the model for training)
# =============================================================================
model.compile(optimizer = 'adam', loss='mean_squared_error')
# metrics = ['rmse']

'''
Note: To evaluate the accuracy one could of course use metrics = ['acc'].
This, however, returns the accuracy score 0.0000.
Why? The reason ist simply, the algorithm wasn't able to explain the data floats
100%. It is simply the wrong measure. The accuracy measure is rather applictable
for categorial data, to measure the accuracy of classifications. 
Here, instead, we'll use the root mean squared error (rmse) which is a customized
metric.


# =============================================================================
# Evaluate
# =============================================================================

# scores = model.evaluate(x_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
'''
# Train the model (Fit)
model.fit(x_train, y_train, batch_size=1, epochs=1)


# Creates a HDF5 file 'my_model.h5'
model.save('my_model.h5')

# from keras.models import load_model

# # # Returns a compiled model identical to the previous one
# model = load_model('D:/Dropbox/Work/Python Scripts/MyProjects/Berkshire_LSTM/my_model.h5')
# =============================================================================
# Validation
# =============================================================================

# Create the testing data set

# Create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len - ac:,:]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset_diff[training_data_len:,:]
for i in range(ac, len(test_data)):
    x_test.append(test_data[i-ac:i,0])
    
# Convert the data to a numpy array (to use it in the LSTM model)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
# 1 since just one close price

# Get the models predicted price values
predictions = model.predict(x_test)
# unscale
predictions = scaler.inverse_transform(predictions)

'''
# Get the root mean squared error (RMSE)

RMSE: good measure of how accurate the model predicts the responds,
what is the standard deviation of the residuals?
The lower the values are, the better is the fit
'''
rmse = np.sqrt(np.mean(predictions - y_test)**2) # ca. 0.7
# resid_I0 = predictions - y_test

resid_I1 = predictions - y_test
# plt.plot(resid_I1)

# =============================================================================
# Ljung-Box test of autocorrelation in residuals
# =============================================================================
# Validate the model using a Ljung-Box test, is there any seriel correlation in
# the residuals? If yes, a K-fold cross-validation may be biased
nlags = 10
ljb_test_I1 = sm.stats.acorr_ljungbox(resid_I1, nlags)
# ljb_test_data = sm.stats.acorr_ljungbox(dataset, nlags)


# 1. lb_stat
# 2. p_value
'''
H0: no autocorrelation in the residuals
H1: Autocorrelation

If the p value is greater than 0.05 then the residuals are independent which 
we want for the model to be correct
The distribution is chi_square
--> we reject the null for sufficiently small p-values, p < alpha=0.05

'''


# =============================================================================
# Plot
# =============================================================================
train = data[:training_data_len]
valid = data[training_data_len:]


# =============================================================================
# Transform back from I(1) to I(0)
# =============================================================================
Prediction = pd.DataFrame(predictions, columns=['I0'])
prediction_I0 = np.squeeze(dataset[training_data_len:])+ Prediction.I0.shift(1)
prediction_I0[0,0] = data.Close[training_data_len]
# predI0 = dataset_diff[training_data_len:] + predictions
valid['Predictions'] = prediction_I0.values


plt.figure(figsize=(8,4))
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price ($)', fontsize=12)
plt.plot(train['Close'], lw=1)
plt.plot(valid[['Close', 'Predictions']], lw=1)
plt.legend(['Train', 'Val', 'Predictions'],loc = 'lower right')
plt.savefig('train_test_BH_I1.eps',  bbox_inches='tight', format='eps')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()


plt.figure(figsize=(8,4))
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price ($)', fontsize=12)
# plt.plot(train['Close'], lw=1)
plt.plot(valid[['Close', 'Predictions']], lw=1)
plt.legend(['Validated', 'Predictions'],loc = 'lower right')
plt.savefig('valid_I1.eps',  bbox_inches='tight', format='eps')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()





# =============================================================================
# Predict tomorrows price
# =============================================================================
# Once again, get the quote
BH_quote = df
# Create a new dataframe
df_new = BH_quote.filter(['Close'])
# Get the last 60 day closing values and convert the dataframe to an array
# last_60_day = df_new [-ac:].values

'''
Compute first differences with numpy
'''
# last_60_day_diff = np.diff(np.squeeze(last_60_day))
last_60_day_diff = df_new.Close.diff()
last_60_day_diff = last_60_day_diff[-ac:] 
last_60_day_diff = np.array(last_60_day_diff)
# last_60_day_diff = np.squeeze(last_60_day_diff)
last_60_day_diff= last_60_day_diff.reshape(-1,1)

# Scale the data to be values between 0 and 1

last_60_days_scaled = scaler.transform(last_60_day_diff)
'''
Notice, I use a variable from further up called 'scaler' 
I'm not gonna using fit.transform since the variables should be
scaled using around the same mean with the same variance than above
'''
# Create an empty list
X_test = []
# Append the pat 60 days
X_test.append(last_60_days_scaled)
# Convert the X_test data set to a numpy array
X_test = np.array(X_test)
# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



# =============================================================================
# Get the predicted scaled price
# =============================================================================
pred_price_I1 = model.predict(X_test)

# Scale back
pred_price_I1 = scaler.inverse_transform(pred_price_I1)

pred_price_I0 = df_new.Close[-1] + pred_price_I1


# =============================================================================
# Is that realistic? How good is the fit?
# =============================================================================

final_close = web.DataReader('BRK-B', data_source='yahoo', start= '2020-02-27',
                    end = '2020-02-27')

final_close = np.array(final_close['Close'])[0]

abs_deviation_prediction = np.abs(pred_price_I0 - final_close)
print('The absolute deviation of the predicted \n to the' 
      ' actual close price is:')
print(abs_deviation_prediction, 'in $') # ca.8