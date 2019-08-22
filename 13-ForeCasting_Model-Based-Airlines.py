'''                       MODEL Based Approach                 '''

# Reset the console
%reset -f

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar

# Load the data
dataset = pd.read_excel('Airlines+Data.xlsx')
dataset.head()
dataset.columns
dataset.dtypes

dataset.isnull().sum() # No missing vlaues
dataset.shape
dataset.drop_duplicates(keep='first', inplace=True) # No duplicates
# get the month from timestamp
dataset['month'] = pd.DatetimeIndex(dataset['Month']).month

# Convert the month number to month name
for i in range(96):
    mon = dataset['month'][i]
    dataset['month'][i] = calendar.month_abbr[mon]
    
# Histogram 
plt.hist(dataset['Passengers'], color='teal');plt.title('Histogram of Passengers')
# Boxplot
import seaborn as sns
sns.boxplot(x='month', y='Passengers', data=dataset).set_title('Boxplot of Passengers')

# Creating dummy variables for month column
mon_dum = pd.DataFrame(pd.get_dummies(dataset['month']))
# Combine the month dummy variables to original dataset
new_dataset = pd.concat([dataset,mon_dum], axis=1)

# Drop the "Month" column
new_dataset.drop(['Month','month'], axis=1, inplace=True)

# Calculating the "t", "t_sqr" and "log_sales"
new_dataset['t'] = np.arange(1,97)
new_dataset['t_sqr'] = new_dataset['t']**2
new_dataset['log_passengers'] = np.log(new_dataset['Passengers'])

# Time Series Plot
plt.plot(new_dataset['Passengers']);plt.title('Time Series Plot of Passengers')

# Splitting the data into trainset and testset
train = new_dataset.head(60)
test = new_dataset.tail(36)

# Fitting the models on datasets
import statsmodels.formula.api as smf

''' ####################### LINEAR ########################## '''

linear_model = smf.ols('Passengers~t',data=train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_linear))**2))
rmse_linear # 44.35

''' ##################### Exponential ############################## '''

Exp = smf.ols('log_passengers~t',data=train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp # 42.91

''' #################### Quadratic ############################### '''

Quad = smf.ols('Passengers~t+t_sqr',data=train).fit()
pred_Quad = pd.Series(Quad.predict(test[["t","t_sqr"]]))
rmse_Quad = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad # 40.31

''' ################### Additive seasonality ######################## '''

add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea # 119.97

''' ################## Additive Seasonality Quadratic ############################ '''

add_sea_Quad = smf.ols('Passengers~t+t_sqr+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_sqr']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad  # 25.49

''' ################## Multiplicative Seasonality ################## '''

Mul_sea = smf.ols('log_passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea # 123.66

''' ##################Multiplicative Additive Seasonality ########### '''

Mul_Add_sea = smf.ols('log_passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea  # 26.10

''' ################## Testing ####################################### '''
# Creating a RMSE dataframe

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
RMSE_table = pd.DataFrame(data)
RMSE_table

# so rmse_add_sea_quad has the least value among the models.

add_sea_quad_error = np.array(test['Passengers']) - np.array(np.exp(pred_add_sea_quad))
add_sea_quad_MAPE = np.mean(np.abs(add_sea_quad_error/(np.array(test['Passengers'])))) *100
print(add_sea_quad_MAPE) # 2.267


# Predicting new values

# ACF plots
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models

tsa_plots.plot_acf(new_dataset['Passengers'], lags=12)
tsa_plots.plot_pacf(new_dataset['Passengers'], lags=12)


# To perform you know order of AR and MA 

# AR models for forecasting errors
from statsmodels.tsa.arima_model import ARIMA
passengers = new_dataset['Passengers']
# from the above plots, both shows a drop-off at the same point, A good starting point for P and Q also 1. 
# d = 0 but for model stability extend to 1
model = ARIMA(new_dataset['Passengers'], order = (1,1,1)).fit(transparams=True)
#help(ARIMA)

# Forecast next 12 months
Arima_forecast = model.forecast(steps=12)[0]
