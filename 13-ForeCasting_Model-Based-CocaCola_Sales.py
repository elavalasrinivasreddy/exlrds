# Reset the console
%reset -f

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar

# Load the data
cocacola = pd.read_excel('CoCaCola_Sales_Rawdata.xlsx')
cocacola.head()
cocacola.shape
cocacola.isnull().sum() # No missing values
cocacola.drop_duplicates(keep='first', inplace=True) # No duplicates
cocacola.columns

#Histogram
plt.hist(cocacola['Sales'], color='teal');plt.title('Histogram of Sales');plt.xlabel('No.of Sales');plt.ylabel('Frequency')
# Boxplot
import seaborn as sns
sns.boxplot(cocacola['Sales'], orient='v', color='orange').set_title('Boxplot of Sales')
sns.boxplot(x='Quarter', y='Sales', data=cocacola)

# Creating two empty lists 
month = []
sales = []

# Convert the Quarter data to monthly data
for i in range(42):
    mon = cocacola['Quarter'][i][0:2]
    if mon == "Q1":
        for x in range(1,4):
            mon_name = calendar.month_abbr[x]
            month.append(mon_name)
            sales.append((cocacola['Sales'][i])/3)
        
    elif mon == "Q2":
        for x in range(4,7):
            mon_name = calendar.month_abbr[x]
            month.append(mon_name)
            sales.append((cocacola['Sales'][i])/3)
            
    elif mon == "Q3":
        for x in range(7,10):
            mon_name = calendar.month_abbr[x]
            month.append(mon_name)
            sales.append((cocacola['Sales'][i])/3)
            
    else:
        for x in range(10,13):
            mon_name = calendar.month_abbr[x]
            month.append(mon_name)
            sales.append((cocacola['Sales'][i])/3)
        
        
#  Creating the new dataframe
dataset = pd.concat([pd.Series(month), pd.Series(sales)], axis=1)
dataset.columns = ['months', 'Sales']

# Boxplot
sns.boxplot(x='months', y='Sales', data=dataset).set_title('Boxplot of Sales(Monthly)')

# Create dummy variables for months
mon_dumm = pd.DataFrame(pd.get_dummies(dataset['months']))
# Combine with original data
new_dataset = pd.concat([dataset, mon_dumm], axis=1)
# Drop the columns
new_dataset.drop('months', axis=1, inplace=True)

# Calculate the "t", "t-sqr" and "log_sales" values
new_dataset['t'] = np.arange(1,127)
new_dataset['t_sqr'] = new_dataset['t']**2
new_dataset['log_sales'] = np.log(new_dataset['Sales'])

# Time series plot
plt.plot(new_dataset['Sales']);plt.title('Time Series Plot for Sales')

# Splitting the data into trainset and testset
train = new_dataset.head(80)
test = new_dataset.tail(46)

# Fitting the models on data
import statsmodels.formula.api as smf

''' ####################### LINEAR ########################## '''

linear_model = smf.ols('Sales~t',data=train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_linear))**2))
rmse_linear # 255.01

''' ##################### Exponential ############################## '''

Exp = smf.ols('log_sales~t',data=train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp # 203.10

''' #################### Quadratic ############################### '''

Quad = smf.ols('Sales~t+t_sqr',data=train).fit()
pred_Quad = pd.Series(Quad.predict(test[["t","t_sqr"]]))
rmse_Quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad # 320.92

''' ################### Additive seasonality ######################## '''

add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea # 582.91

''' ################## Additive Seasonality Quadratic ############################ '''

add_sea_Quad = smf.ols('Sales~t+t_sqr+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_sqr']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad  # 290.91

''' ################## Multiplicative Seasonality ################## '''

Mul_sea = smf.ols('log_sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea # 594.35

''' ##################Multiplicative Additive Seasonality ########### '''

Mul_Add_sea = smf.ols('log_sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea  # 179.90

''' ################## Testing ####################################### '''
# Creating a RMSE dataframe

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
RMSE_table = pd.DataFrame(data)
RMSE_table

# so rmse_Multi_add_sea has the least value among the models.

multi_error = np.array(test['Sales']) - np.array(np.exp(pred_Mult_add_sea))
multi_MAPE = np.mean(np.abs(multi_error/(np.array(test['Sales'])))) *100
print(multi_MAPE) # 11.63

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
model = ARIMA(new_dataset['Sales'], order = (1,1,4)).fit(transparams=True)
#help(ARIMA)

# Forecast next 12 months
Arima_forecast = model.forecast(steps=12)[0]
