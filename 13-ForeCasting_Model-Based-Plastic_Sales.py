'''                 Model Based Approach                  '''

# Reset the Console
%reset -f

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
dataset = pd.read_csv('PlasticSales.csv')
dataset.head()
dataset.columns
dataset.dtypes
dataset.isnull().sum() # No missing values
dataset.shape
dataset.drop_duplicates(keep='first', inplace=True) # No duplicates

month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
p = dataset['Month'][0]
p[0:3]

# Creating a new months column
dataset['months'] = 0
for i in range(60):
    p = dataset['Month'][i]
    dataset['months'][i] = p[0:3]

# Histogram 
plt.hist(dataset['Sales'], color='coral');plt.title('Histogram of Sales')

# Boxplot
import seaborn as sns
sns.boxplot(x = 'months', y='Sales', data=dataset).set_title('Boxplot of Sales')

# Creating dummy variables for months column
mon_dum = pd.DataFrame(pd.get_dummies(dataset['months']))
# Combine the month dummy variables to original dataset
new_dataset = pd.concat([dataset,mon_dum], axis=1)

# Drop the "Month" column
new_dataset.drop('Month', axis=1, inplace=True)

# Calculating the "t", "t_sqr" and "log_sales"
new_dataset['t'] = np.arange(1,61)
new_dataset['t_sqr'] = new_dataset['t']**2
new_dataset['log_sales'] = np.log(new_dataset['Sales'])

# Time Series Plot
plt.plot(new_dataset['Sales']);plt.title('Time Series Plot of Sales')

# Splitting the data into trainset and testset
train = new_dataset.head(45)
test = new_dataset.tail(15)

# Fitting the models on datasets
import statsmodels.formula.api as smf

''' ####################### LINEAR ########################## '''

linear_model = smf.ols('Sales~t',data=train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_linear))**2))
rmse_linear # 241.36

''' ##################### Exponential ############################## '''

Exp = smf.ols('log_sales~t',data=train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp # 247.90

''' #################### Quadratic ############################### '''

Quad = smf.ols('Sales~t+t_sqr',data=train).fit()
pred_Quad = pd.Series(Quad.predict(test[["t","t_sqr"]]))
rmse_Quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad # 276.16

''' ################### Additive seasonality ######################## '''

add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea # 257.13

''' ################## Additive Seasonality Quadratic ############################ '''

add_sea_Quad = smf.ols('Sales~t+t_sqr+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_sqr']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad  # 151.16

''' ################## Multiplicative Seasonality ################## '''

Mul_sea = smf.ols('log_sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea # 261.53

''' ##################Multiplicative Additive Seasonality ########### '''

Mul_Add_sea = smf.ols('log_sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea  # 133.29

''' ################## Testing ####################################### '''
# Creating a RMSE dataframe

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
RMSE_table = pd.DataFrame(data)
RMSE_table

# so rmse_Multi_add_sea has the least value among the models.

multi_error = np.array(test['Sales']) - np.array(np.exp(pred_Mult_add_sea))
multi_MAPE = np.mean(np.abs(multi_error/(np.array(test['Sales'])))) *100
print(multi_MAPE) # 8.67



# Predicting new values

# ACF plots
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models

tsa_plots.plot_acf(new_dataset['Sales'], lags=12)
tsa_plots.plot_pacf(new_dataset['Sales'], lags=12)


# To perform you know order of AR and MA 

# AR models for forecasting errors
from statsmodels.tsa.arima_model import ARIMA
sales = new_dataset['Sales']
# from the above plots, both shows a drop-off at the same point, A good starting point for P and Q also 1. 
# d = 0 but for model stability extend to 1
model = ARIMA(new_dataset['Sales'], order = (1,1,1)).fit(transparams=True)
#help(ARIMA)

# Forecast next 12 months
Arima_forecast = model.forecast(steps=12)[0]

# Residuals
#AR_resid = pd.DataFrame(resid_AR.resid)
#
# ACF plot for residuals 
#tsa_plots.plot_acf(AR_resid, lag=12)