# Reset the console
%reset -f

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
dataset = pd.read_csv('Computer_Data.csv', index_col=0)
dataset.head()
dataset.info()
dataset.shape
dataset.dtypes

# Drop duplicate rows if any
dataset = dataset.drop_duplicates(keep='first')# removed 76 duplicate rows
dataset.shape

# Missing Values
dataset.isnull().sum() # No missing values
dataset.columns
dataset['cd'].value_counts()
dataset['multi'].value_counts()
dataset['premium'].value_counts()

# Statistical description
dataset.describe()

# Measures of Dispersion
np.var(dataset)
np.std(dataset)

# Skewness and Kurtosis
from scipy.stats import skew, kurtosis, norm
skew(dataset.drop(['cd','multi', 'premium'], axis=1))
kurtosis(dataset.drop(['cd','multi', 'premium'], axis=1))

# Histogram
plt.hist(dataset['price']);plt.xlabel('Price');plt.ylabel('Frequency');plt.title('Histogram of Price')

plt.hist(dataset['speed'], color='coral');plt.xlabel('Speed');plt.ylabel('Frequency');plt.title('Histogram of Speed')

plt.hist(dataset['hd'], color='skyblue');plt.xlabel('HD');plt.ylabel('Frequency');plt.title('Histogram of HD')

plt.hist(dataset['ram'], color='orange');plt.xlabel('RAM');plt.ylabel('Frequency');plt.title('Histogram of RAM')

plt.hist(dataset['screen'], color='lightgreen');plt.xlabel('Screen');plt.ylabel('Frequency');plt.title('Histogram of Screen')

plt.hist(dataset['ads'], color='brown');plt.xlabel('Ads');plt.ylabel('Frequency');plt.title('Histogram of Ads')

plt.hist(dataset['trend'], color='violet');plt.xlabel('Trend');plt.ylabel('Frequency');plt.title('Histogram of Trend')

# Barplot for Categorical data
import seaborn as sns
sns.countplot(x='cd', data=dataset).set_title('Count plot of CD')
sns.countplot(x='multi', data=dataset).set_title('Countplot of Multi')
sns.countplot(x='premium',hue='screen', data=dataset).set_title('Count plot of Premium')

# # Normal Q-Q plot
plt.plot(dataset.drop(['cd', 'multi', 'premium'], axis=1), alpha=1); plt.legend(['price', 'speed', 'hd', 'ram', 'screen', 'ads', 'trend'])

price = np.array(dataset['price'])
speed = np.array(dataset['speed'])
hd = np.array(dataset['hd'])
ram = np.array(dataset['ram'])
screen = np.array(dataset['screen'])
ads = np.array(dataset['ads'])
trend = np.array(dataset['trend'])

from scipy import stats
stats.probplot(price, dist='norm', plot=plt);plt.title('Q-Q plot of Price')
stats.probplot(speed, dist='norm', plot=plt);plt.title('Q-Q plot of Speed')
stats.probplot(hd, dist='norm', plot=plt);plt.title('Q-Q plot of HD')
stats.probplot(ram, dist='norm', plot=plt);plt.title('Q-Q plot of RAM')
stats.probplot(screen, dist='norm', plot=plt);plt.title('Q-Q plot of Screen')
stats.probplot(ads, dist='norm', plot=plt);plt.title('Q-Q plot of Ads')
stats.probplot(trend, dist='norm', plot=plt);plt.title('Q-Q plot of Trend')

# Normal probability distribution
# Price
x_price = np.linspace(np.min(price), np.max(price))
y_price = stats.norm.pdf(x_price, np.mean(x_price), np.std(x_price))
plt.plot(x_price, y_price);plt.xlim(np.min(x_price), np.max(x_price));plt.xlabel('Price');plt.ylabel('Probability');plt.title('Normal Probability Distribution of Price')

# Speed
x_speed = np.linspace(np.min(speed), np.max(speed))
y_speed = stats.norm.pdf(x_speed, np.mean(x_speed), np.std(x_speed))
plt.plot(x_speed, y_speed, color='coral');plt.xlim(np.min(x_speed), np.max(x_speed));plt.xlabel('Speed');plt.ylabel('Probability');plt.title('Normal Probability Distribution of Speed')

# HD
x_hd = np.linspace(np.min(hd), np.max(hd))
y_hd = stats.norm.pdf(x_hd, np.mean(x_hd), np.std(x_hd))
plt.plot(x_hd, y_hd, color='skyblue');plt.xlim(np.min(x_hd), np.max(x_hd));plt.xlabel('HD');plt.ylabel('Probability');plt.title('Normal Probability Distribution of HD')

# RAM
x_ram = np.linspace(np.min(ram), np.max(ram))
y_ram = stats.norm.pdf(x_ram, np.mean(x_ram), np.std(x_ram))
plt.plot(x_ram, y_ram, color='orange');plt.xlim(np.min(x_ram), np.max(x_ram));plt.xlabel('RAM');plt.ylabel('Probability');plt.title('Normal Probability Distribution of RAM')

# Screen
x_screen = np.linspace(np.min(screen), np.max(screen))
y_screen = stats.norm.pdf(x_screen, np.mean(x_screen), np.std(x_screen))
plt.plot(x_screen, y_screen, color='brown');plt.xlim(np.min(x_screen), np.max(x_screen));plt.xlabel('Screen');plt.ylabel('Probability');plt.title('Normal Probability Distribution of Screen')

# Ads
x_ads = np.linspace(np.min(ads), np.max(ads))
y_ads = stats.norm.pdf(x_ads, np.mean(x_ads), np.std(x_ads))
plt.plot(x_ads, y_ads, color='lightgreen');plt.xlim(np.min(x_ads), np.max(x_ads));plt.xlabel('Ads');plt.ylabel('Probability');plt.title('Normal Probability Distribution of Ads')

# Trend
x_trend = np.linspace(np.min(trend), np.max(trend))
y_trend = stats.norm.pdf(x_trend, np.mean(x_trend), np.std(x_trend))
plt.plot(x_trend, y_trend, color='violet');plt.xlim(np.min(x_trend), np.max(x_trend));plt.xlabel('Trend');plt.ylabel('Probability');plt.title('Normal Probability Distribution of Trend')

# Boxplot
sns.boxplot(data=dataset).set_title('Boxplot of Variables')
sns.distplot(dataset['price'], fit=stats.norm, kde=False, color='coral')

# Scatterplot
sns.scatterplot(x='hd', y='trend', data=dataset).set_title('Scatterplot of HD and Trend')
sns.scatterplot(x='ram', y='price', data=dataset).set_title('Scatterplot of RAM and Price')
sns.scatterplot(x='ram', y='hd', data=dataset).set_title('Scatterplot of RAM and HD')
sns.scatterplot(x='hd', y='price', data=dataset).set_title('Scatterplot of HD and Price')

sns.pairplot(dataset)
sns.pairplot(dataset, diag_kind='kde')
sns.pairplot(dataset, hue='premium')
sns.pairplot(dataset, kind='reg')

# Heatmap
dataset.corr().round(2)
sns.heatmap(dataset.corr(), annot=True)

# Detect and remiove outliers
num_data = dataset.drop(['cd','multi','premium'], axis=1)
# Z score
from scipy import stats
Z = np.abs(stats.zscore(num_data))
print(Z)
threshold = 3
print(np.where(Z>3))
print(Z[410][0])

df_out = num_data[(Z<3).all(axis=1)] # 137 rows removed [6046 rows]
df_out.shape

'''# Find the Outliers "Tukey IQR"
def find_outliers(x):
	Q1 = np.percentile(x, 25)
	Q3 = np.percentile(x, 75)
	IQR = Q3 - Q1
	floor = Q1 - 1.5*IQR
	ceiling = Q3 + 1.5*IQR
	outlier_indices = list(x.index[(x<floor) | (x>ceiling)])
	outlier_values = list(x[outlier_indices])
	return outlier_indices, outlier_values

outlier_indices, outlier_values = find_outliers(df_out['price'])
#print(np.sort(outlier_values))
df_out = df_out.drop(outlier_indices, axis=0)
df_out.shape # [6010 rows]

outlier_indices, outlier_values = find_outliers(df_out['speed'])
df_out = df_out.drop(outlier_indices, axis=0)
df_out.shape # [6010 rows]

outlier_indices, outlier_values = find_outliers(df_out['hd'])
df_out = df_out.drop(outlier_indices, axis=0)
df_out.shape # [5628 rows] 382 rows removed

outlier_indices, outlier_values = find_outliers(df_out['ram'])
df_out = df_out.drop(outlier_indices, axis=0)
df_out.shape # [4752 rows] 876 rows removed

outlier_indices, outlier_values = find_outliers(df_out['screen'])
df_out = df_out.drop(outlier_indices, axis=0)
df_out.shape # [4313 rows] 439 rows removed

outlier_indices, outlier_values = find_outliers(df_out['price'])
df_out = df_out.drop(outlier_indices, axis=0)
df_out.shape # [4229 rows] 84 rows removed

outlier_indices, outlier_values = find_outliers(df_out['hd'])
df_out = df_out.drop(outlier_indices, axis=0)
df_out.shape # [4180 rows] 49 rows removed

outlier_indices, outlier_values = find_outliers(df_out['price'])
df_out = df_out.drop(outlier_indices, axis=0)
df_out.shape # [4175 rows] 5 rows are removed

sns.boxplot(data=df_out) # Total 2008 rows are removed '''

# Categorical data
cat_data = dataset.loc[:, ['cd','multi','premium']]
cat_data = pd.get_dummies(cat_data, prefix=['cd','multi','premium'], drop_first=True)

# concatenate the df_out and cat_data
final_data = pd.concat([df_out, cat_data], axis=1)
final_data.isnull().sum()
final_data.dropna(axis=0, inplace=True)

# Metric features
X = final_data.iloc[:,1:].values
Y = final_data.iloc[:, 0].values


# Splitting the data into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

# Build the Multi linear regressor model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=True)
regressor.fit(X_train, Y_train)

# Prediction on train set
reg_train_pred = regressor.predict(X_train)
# train set residuals
reg_train_resid = reg_train_pred - Y_train
# RMSE of train set
rmse = np.sqrt(np.mean(reg_train_resid**2))
print(rmse) # 232.25

# Prediction on test set
reg_test_pred = regressor.predict(X_test)
# test set residuals
reg_test_resid = reg_test_pred - Y_test
# RMSE of test set
rmse = np.sqrt(np.mean(reg_test_resid**2))
print(rmse) # 254.61

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as smf
X = np.append(arr=np.ones((6046,1)).astype(int), values= X, axis=1) # Adding 1's matrix to X

# With all the independent variables
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9]]
model1 = smf.OLS(endog = Y, exog = X_opt).fit()
model1.summary() # R^2 = 0.779 and Adj.R^2 = 0.779 and all are significant 
model1.params
# Confidence value 99%
print(model1.conf_int(0.01))

'''# Remove X8
X_opt = X[:, [0,1,2,3,4,5,6,7,9]]
model2 = smf.OLS(endog = Y, exog = X_opt).fit()
model2.summary() # R^2 = 1 and Adj.R^2 = 1
model2.params
# Confidence value 99%
print(model2.conf_int(0.01))

# Remove X9
X_opt = X[:, [0,1,2,3,4,5,6,7]]
model3 = smf.OLS(endog = Y, exog = X_opt).fit()
model3.summary() # R^2 = 1 and Adj.R^2 = 1
model3.params
# Confidence value 99%
print(model3.conf_int(0.01))

# Remove X4
X_opt = X[:, [0,1,2,3,5,6,7]]
model4 = smf.OLS(endog = Y, exog = X_opt).fit()
model4.summary() # R^2 = 1 and Adj.R^2 = 1
model4.params
# Confidence value 99%
print(model4.conf_int(0.01))

# Remove X6
X_opt = X[:, [0,1,2,3,5,6]]
model5 = smf.OLS(endog = Y, exog = X_opt).fit()
model5.summary() # R^2 = 1 and Adj.R^2 = 1
# All independent variables are significant
model5.params
# Confidence value 99%
print(model5.conf_int(0.01)) '''

# Check data have any influential values
# Influence index plots
import statsmodels.api as sm
sm.graphics.influence_plot(model1) 
# index 4287,3877,3977,74,296,300,4582,197,213,220,51,166,107,4520,1448,1431,3,4567,1085,3953,3911,4243

new_X = np.delete(X, [4287,3877,3977,4582,74,51,220,296,300,197,213,166,107,4520,1448,1431,3,4567,1085,3953,3911,4243], axis=0)
new_Y = np.delete(Y, [4287,3877,3977,4582,74,51,220,296,300,197,213,166,107,4520,1448,1431,3,4567,1085,3953,3911,4243], axis=0)

# index 41, 173,131,65,3,60,2051,3210,3079,3066,3373,1582,3802,3703,3373,1582,1737,3805,3750,3845,3628,3727,121,1162,
#56,3148,3066,3440,2825,3277
#new_X = np.delete(X, [41, 173,131,65,3,60,2051,3210,3079,3066,3373,1582,3802,3703,3373,1582,1737,3805,3750,3845,3628,3727,121,1162,56,3148,3066,3440,2825,3277], axis=0)
#new_Y = np.delete(Y, [41, 173,131,65,3,60,2051,3210,3079,3066,3373,1582,3802,3703,3373,1582,1737,3805,3750,3845,3628,3727,121,1162,56,3148,3066,3440,2825,3277], axis=0)


# Preparing new model
new_X_opt = new_X[:, [0,1,2,3,4,5,6,7,8,9]]
model1_new = smf.OLS(endog=new_Y, exog=new_X_opt).fit()
model1_new.summary() # R^2 = 0.787 & Adj.R^2 = 0.787  All are significant
model1_new.params
print(model1_new.conf_int(0.01)) 

'''# Remove X5
new_X_opt = new_X[:, [0,1,2,3,5]]
model2_new = smf.OLS(endog=new_Y, exog=new_X_opt).fit()
model2_new.summary()

# Remove X3
new_X_opt = new_X[:, [0,1,2,5]]
model3_new = smf.OLS(endog=new_Y, exog=new_X_opt).fit()
model3_new.summary() # All variables are significant
model3_new.params
print(model3_new.conf_int(0.01)) '''

# Calculating VIF's values of independent variables
rsq_speed = smf.OLS(endog=X[:,1], exog=X[:, [0,2,3,4,5,6,7,8,9]]).fit().rsquared
vif_speed = 1/(1-rsq_speed) # 8.73 | 1.24
print(vif_speed)

rsq_hd = smf.OLS(endog=X[:,2], exog=X[:, [0,1,3,4,5,6,7,8,9]]).fit().rsquared
vif_hd = 1/(1-rsq_hd) # 17.82 | 4.51
print(vif_hd)

rsq_ram = smf.OLS(endog=X[:,3], exog=X[:, [0,1,2,4,5,6,7,8,9]]).fit().rsquared
vif_ram = 1/(1-rsq_ram) # 9.95 | 3.04
print(vif_ram)

rsq_screen = smf.OLS(endog=X[:,4], exog=X[:, [0,1,2,3,5,6,7,8,9]]).fit().rsquared
vif_screen = 1/(1-rsq_screen) # 35.89 | 1.08
print(vif_screen)

rsq_ads = smf.OLS(endog=X[:,5], exog=X[:, [0,1,2,3,4,6,7,8,9]]).fit().rsquared
vif_ads = 1/(1-rsq_ads) # 11.32 | 1.18
print(vif_ads)

rsq_trend = smf.OLS(endog=X[:,6], exog=X[:, [0,1,2,3,4,5,7,8,9]]).fit().rsquared
vif_trend = 1/(1-rsq_trend) # 10.71 | 2.078
print(vif_trend)

rsq_cd = smf.OLS(endog=X[:,7], exog=X[:, [0,1,2,3,4,5,6,8,9]]).fit().rsquared
vif_cd = 1/(1-rsq_cd) # 3.44 | 1.88
print(vif_cd)

rsq_multi = smf.OLS(endog=X[:,8], exog=X[:, [0,1,2,3,4,5,6,7,9]]).fit().rsquared
vif_multi = 1/(1-rsq_multi) # 1.50 | 1.37
print(vif_multi)

rsq_premium = smf.OLS(endog=X[:,9], exog=X[:, [0,1,2,3,4,5,6,7,8]]).fit().rsquared
vif_premium = 1/(1-rsq_premium) # 10.63 | 1.14
print(vif_premium)

# Storing VIF values in a dataframe
d1 = {'variables':['speed', 'hd', 'ram', 'screen', 'ads', 'trend', 'cd', 'multi', 'premium'], 'VIF':[vif_speed,vif_hd,vif_ram,vif_screen,vif_ads,vif_trend,vif_cd,vif_multi,vif_premium]}
vif_dataframe = pd.DataFrame(d1)
vif_dataframe # Screen having higher VIF value so exclude in the prediction

# Added Variable Plot
sm.graphics.plot_partregress_grid(model1_new)
# All are vary with target variable 
# model1_new is the final model
new_Y_pred = model1_new.predict(new_X)
new_Y_pred

# Linearity
# Observed values VS fitted values
plt.scatter(new_Y, new_Y_pred, c='r');plt.xlabel('Observed_Vlaues');plt.ylabel('Fitted_Values')

# Normality plot for residuals
# Histogram
plt.hist(model1_new.resid_pearson) # Checking the standardized residuals are normally distributed

# Q-Q plot
stats.probplot(model1_new.resid_pearson, dist='norm', plot=plt)

# Homoscedasticity
# Residuals VS fitted values
plt.scatter(new_Y_pred, model1_new.resid_pearson, c='r'), plt.axhline(y=0, color='blue');plt.xlabel('Fitted_Values');plt.ylabel('Residuals')

# Splitting the new data into train set and test set
newX_train, newX_test, newY_train, newY_test = train_test_split(new_X, new_Y, test_size=0.20, random_state=0)

# Prepare the model on the train set data
final_model = smf.OLS(endog=newY_train, exog=newX_train).fit()

# Train data prediction
train_pred = final_model.predict(newX_train)

# train residual values
train_resid = train_pred - newY_train

# RMSE value of train data
train_rmse = np.sqrt(np.mean(train_resid**2))
print(train_rmse) # 252.75

# prediction on test set data
test_pred = final_model.predict(newX_test)

# Test set residual values
test_resid = test_pred - newY_test

# RMSE value for test data
test_rmse = np.sqrt(np.mean(test_resid**2))
print(test_rmse) # 254.02
