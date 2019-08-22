# Reset the console
%reset -f

# Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the data
dataset = pd.read_csv('50_Startups.csv')

# Change the columns names
list(dataset)
dataset.columns = ['r&d_spend', 'adminstration', 'marketing_spend', 'state', 'profit']

dataset.head(10)
dataset.info()
dataset.describe().round(2)
dataset.shape
dataset.dtypes

# Drop duplicate rows if any
dataset = dataset.drop_duplicates(keep='first')
dataset.shape

# Missing Values
dataset.isnull().sum()
dataset['state'].value_counts()

# Measures of Dispersion
np.var(dataset).round(2)
np.std(dataset).round(2)

# Skewness and Kurtosis
from scipy.stats import skew, kurtosis, norm
skew(dataset.drop('state', axis=1)).round(2)
kurtosis(dataset.drop('state', axis=1)).round(2)

# Histogram
plt.hist(dataset['r&d_spend'], bins=10, color='coral',density=True);plt.xlabel('R&D_spend', fontsize=15);plt.ylabel('Frequency', fontsize=15);plt.title('Histogram of R&D_spend')

plt.hist(dataset['adminstration'], bins=10, color='skyblue',density=True);plt.xlabel('Adminstration', fontsize=15);plt.ylabel('Frequency', fontsize=15);plt.title('Histogram of Adminstration')

plt.hist(dataset['marketing_spend'], bins=10, color='orange',density=True);plt.xlabel('Marketing_spend', fontsize=15);plt.ylabel('Frequency', fontsize=15);plt.title('Histogram of Marketing_spend')

plt.hist(dataset['profit'], bins=10, color='lightgreen',density=True);plt.xlabel('Profit', fontsize=15);plt.ylabel('Frequency',fontsize=15);plt.title('Histogram of Profit')

# Bar plot for state
import seaborn as sns
sns.countplot(dataset.state).set_title('Frequency of Each State')

# Normal Q-Q plot
plt.plot(dataset.drop('state', axis=1), alpha=1); plt.legend(['R&D_spend', 'Adminstration', 'Marketing_spend', 'Profit'])

rd = np.array(dataset['r&d_spend'])
ad = np.array(dataset['adminstration'])
mk = np.array(dataset['marketing_spend'])
pf = np.array(dataset['profit'])

from scipy import stats
stats.probplot(rd, dist='norm', plot=plt);plt.title('Q-Q plot of R&D spend')
stats.probplot(ad, dist='norm', plot=plt);plt.title('Q-Q plot of Adminstration')
stats.probplot(mk, dist='norm', plot=plt);plt.title('Q-Q plot of Marketing spend')
stats.probplot(pf, dist='norm', plot=plt);plt.title('Q-Q plot of Profit')

# Normal probability distribution
# R&D Spend
x_rd = np.linspace(np.min(rd), np.max(rd))
y_rd = stats.norm.pdf(x_rd, np.mean(x_rd), np.std(x_rd))
plt.plot(x_rd, y_rd, color='coral');plt.xlim(np.min(x_rd), np.max(x_rd));plt.xlabel('R&D Spend');plt.ylabel('Probability');plt.title('Normal Probability Distribution of R&D Spend')

#Adminstration
x_ad = np.linspace(np.min(ad), np.max(ad))
y_ad = stats.norm.pdf(x_ad, np.mean(x_ad), np.std(x_ad))
plt.plot(x_ad, y_ad, color='skyblue');plt.xlim(np.min(x_ad), np.max(x_ad));plt.xlabel('Adminstration');plt.ylabel('Probability');plt.title('Normal Probability Distribution of Adminstration')

#Marketing Spend
x_mk = np.linspace(np.min(mk), np.max(mk))
y_mk = stats.norm.pdf(x_mk, np.mean(x_mk), np.std(x_mk))
plt.plot(x_mk, y_mk, color='orange');plt.xlim(np.min(x_mk), np.max(x_mk));plt.xlabel('Marketing Spend');plt.ylabel('Probability');plt.title('Normal Probability Distribution of Marketing Spend')

# Profit
x_pf = np.linspace(np.min(pf), np.max(pf))
y_pf = stats.norm.pdf(x_pf, np.mean(x_pf), np.std(x_pf))
plt.plot(x_pf, y_pf, color='brown');plt.xlim(np.min(x_pf), np.max(x_pf));plt.xlabel('Profit');plt.ylabel('Probability');plt.title('Normal Probability Distribution of Profit')

# Boxplot
sns.boxplot(data=dataset).set_title('Boxplot of Variables')
sns.distplot(dataset['profit'], fit=norm, kde=False, color='coral')

# Scatterplot
sns.scatterplot(x='r&d_spend', y='profit', data=dataset).set_title('Scatterplot of R&D and Profit')
sns.scatterplot(x='marketing_spend', y='r&d_spend', data=dataset).set_title('Scatterplot of Marketing Spend and R&D')
sns.scatterplot(x='marketing_spend', y='profit', data=dataset).set_title('Scatterplot of Marketing Spend and Profit')

sns.pairplot(dataset)
sns.pairplot(dataset, diag_kind='kde')
sns.pairplot(dataset, hue='state')
sns.pairplot(dataset, kind='reg')

# Heatmap
dataset.corr().round(2)
sns.heatmap(dataset.corr(), annot=True)

# Metrics features
X = dataset.iloc[:, 0:4].values
Y = dataset.iloc[:,-1].values

# Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
# Dummy encoding
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Creating Dummy variables
#state_dummy = pd.get_dummies(dataset['state'],prefix='state', drop_first=True)

# Avoid Dummy trap
X = X[:,1:]

# Splitting the data into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Concatinating the state dummy and dataset
#new_data = pd.concat([dataset, state_dummy], axis=1)
#new_data.columns=['r&d_spend','adminstration','marketing_spend','state','profit','state_florida','state_newyork']

'''# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) '''

# Fitting Multiple Linear Regression to Train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=True)
regressor.fit(X_train, Y_train)

# Prediction on train set
reg_train_pred = regressor.predict(X_train)
# Train set residuals
reg_train_resid = reg_train_pred - Y_train
# RMSE value of train set
reg_train_rmse = np.sqrt(np.mean(reg_train_resid**2))
print(reg_train_rmse) # 9271.31

# Prediction on test set data
reg_test_pred = regressor.predict(X_test)
# Test set residuals
reg_test_resid = reg_test_pred - Y_test
# RMSE value of test set
reg_test_rmse = np.sqrt(np.mean(reg_test_resid**2))
print(reg_test_rmse) # 8591.23

# Building the Optimal model using Backward Elimination
import statsmodels.formula.api as smf
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)   # Adding X metrics to 1's Column metrix

# With all the independent variables
X_opt = X[:, [0,1,2,3,4,5]] # Metrix containing all the independent variables
model1 = smf.OLS(endog=Y, exog=X_opt).fit()
model1.summary() # R^2 = 0.951 & Adj.R^2 = 0.945
model1.params
# Confidence value 99%
print(model1.conf_int(0.01))

# Remove X2
X_opt = X[:, [0,1,3,4,5]]
model2= smf.OLS(endog=Y, exog=X_opt).fit()
model2.summary() # R^2 = 0.951 & Adj.R^2 = 0.946
model2.params
# Confidence value 99%
print(model2.conf_int(0.01))

# Remove X1
X_opt = X[:, [0,3,4,5]]
model3= smf.OLS(endog=Y, exog=X_opt).fit()
model3.summary() # R^2 = 0.951 & Adj.R^2 = 0.948
model3.params
# Confidence value 99%
print(model3.conf_int(0.01))

# Remove X4
X_opt = X[:, [0,3,5]]
model4= smf.OLS(endog=Y, exog=X_opt).fit()
model4.summary() # R^2 = 0.950 & Adj.R^2 = 0.948  # R^2 decreased compared to model3
model4.params
# Confidence value 99%
print(model4.conf_int(0.01))

# Remove X5
X_opt = X[:, [0,3]]
model5= smf.OLS(endog=Y, exog=X_opt).fit()
model5.summary() # R^2 = 0.947 & Adj.R^2 = 0.945 # Both are decreased compare to model3
model5.params
# Confidence value 99%
print(model5.conf_int(0.01))

# Checking whether data has any influentail values
# Influence index plots
import statsmodels.api as sm
sm.graphics.influence_plot(model1) # index 49,48,46,45 showing high influence so we can exclude that entire row
# Studentized Residuals = Residual/standard deviation of residuals

new_X = np.delete(X, [49,48,46,45], axis=0)
new_X_opt = new_X[:, [0,1,2,3,4,5]]
new_Y = np.delete(Y, [49,48,46,45], axis=0)

# Preapring new model
model1_new = smf.OLS(endog=new_Y, exog=new_X_opt).fit()
model1_new.summary() # R^2 = 0.963 and Adj.R^2 = 0.958
model1_new.params
# Confidence value 99%
print(model1_new.conf_int(0.01))

# Exclude only 49, 48, 45 only
new_X = np.delete(X, [49,48,45], axis=0)
new_X_opt = new_X[:, [0,1,2,3,4,5]]
new_Y = np.delete(Y, [49,48,45], axis=0)

# Preparing new model
model2_new = smf.OLS(endog=new_Y, exog=new_X_opt).fit()
model2_new.summary() # R^2 = 0.965 & Adj.R^2= 0.960
model2_new.params
# confidence value 99%
print(model2_new.conf_int(0.01))

# Exclude only 49,48 only
new_X = np.delete(X, [49,48], axis=0)
new_X_opt = new_X[:, [0,1,2,3,4,5]]
new_Y = np.delete(Y, [49,48], axis=0)

# Preparing new model
model3_new = smf.OLS(endog=new_Y, exog=new_X_opt).fit()
model3_new.summary() # R^2 = 0.963 & Adj.R^2= 0.958 # both are decreased compared to model2_new
model3_new.params
# confidence value 99%
print(model2_new.conf_int(0.01))


# Calculating VIF's values of independent variables
rsq_st1 = smf.OLS(endog=X[:,1], exog=X[:, [0,2,3,4,5]]).fit().rsquared
vif_st1 = 1/(1-rsq_st1) # 1.39

rsq_st2 = smf.OLS(endog=X[:,2], exog=X[:, [0,1,3,4,5]]).fit().rsquared
vif_st2 = 1/(1-rsq_st2) # 1.34

rsq_rd = smf.OLS(endog=X[:,3], exog=X[:, [0,1,2,4,5]]).fit().rsquared
vif_rd = 1/(1-rsq_rd) # 2.5

rsq_ad = smf.OLS(endog=X[:,4], exog=X[:, [0,1,2,3,5]]).fit().rsquared
vif_ad = 1/(1-rsq_ad) # 1.18

rsq_mk = smf.OLS(endog=X[:,5], exog=X[:, [0,1,2,3,4]]).fit().rsquared
vif_mk = 1/(1-rsq_mk) # 2.42

# Sorting VIF's values in dataframe
d1 = {'variables':['r&d_spend', 'adminstration', 'marketing_spend', 'state_florida', 'state_newyork'], 'VIF':[vif_rd, vif_ad, vif_mk, vif_st1, vif_st2]}
vif_frame = pd.DataFrame(d1)
vif_frame
# All variables satisfies the VIF <10

# Added Variable plot
sm.graphics.plot_partregress_grid(model1_new)
# All variables are changing with target variable

# model2_new is the final model and R^2 increased form 0.951 to 0.965
# Exclude only 49, 48, 45 only
new_X = np.delete(X, [49,48,45], axis=0)
new_X_opt = new_X[:, [0,1,2,3,4,5]]
new_Y = np.delete(Y, [49,48,45], axis=0)

new_Y_pred = model2_new.predict(new_X_opt)
new_Y_pred
# Added Variable Plot for final model

# Linearity
# Observed values VS fitted values
plt.scatter(new_Y, new_Y_pred, c='r');plt.xlabel('Observed_Values');plt.ylabel('Fitted_Values')

# Residuals VS fitted values
#plt.scatter(new_Y_pred, model1_new.resid_pearson, c='b'),plt.axhline(y=0, color='red');plt.xlabel('Fitted_Values');plt.ylabel('Residuals')

# Normality plot for Residuals
# Histogram
plt.hist(model2_new.resid_pearson) # Checking the standardized residuals are normallly distributed

# Q-Q plot 
stats.probplot(model2_new.resid_pearson, dist='norm', plot=plt)

# Homoscedasticity
# Residuals VS fitted values
plt.scatter(new_Y_pred, model2_new.resid_pearson, c='r'),plt.axhline(y=0, color='blue');plt.xlabel('Fitted_Values');plt.ylabel('Residuals')

# Splitting the new data into train set and test set
from sklearn.model_selection import train_test_split
newX_train, newX_test, newY_train, newY_test = train_test_split(new_X, new_Y, test_size=0.20, random_state=0)

'''# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
newX_train = sc_X.fit_transform(newX_train)
newX_test = sc_X.transform(newX_test) '''

# Prepare the model on the train set data
final_model = smf.OLS(endog=newY_train, exog=newX_train).fit()

# Train data prediction
train_pred = final_model.predict(newX_train)

# train residual values
train_resid = train_pred - newY_train

# RMSE value of train data
train_rmse = np.sqrt(np.mean(train_resid**2))
print(train_rmse) # 7197.61 | 117118.18(standardized)

# prediction on test set data
test_pred = final_model.predict(newX_test)

# Test set residual values
test_resid = test_pred - newY_test

# RMSE value for test data
test_rmse = np.sqrt(np.mean(test_resid**2))
print(test_rmse) # 5098.10 | 116613.74(standardied)
 
