
# Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
corolla = pd.read_csv('ToyotaCorolla.csv', encoding = 'unicode_escape')
dataset = corolla.loc[:,["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
list(dataset)
dataset.columns = ['price', 'age', 'km', 'hp', 'cc', 'doors', 'gears', 'quarterly_tax', 'weight']

dataset.head()
dataset.info()
dataset.describe().round(2)
dataset.shape
dataset.dtypes

# Drop duplicate rows if any
dataset = dataset.drop_duplicates(keep='first')
dataset.shape

# Missing values
dataset.isnull().sum()

# Measures of Dispersion
np.var(dataset)
np.std(dataset)

# Skewness and kurtosis
from scipy.stats import skew, kurtosis, norm
skew(dataset)
kurtosis(dataset)

# Histogram
plt.hist(dataset['price']);plt.title('Histogram of Price'); plt.xlabel('Price'); plt.ylabel('Frequency')
plt.hist(dataset['age'], color = 'coral');plt.title('Histogram of Age'); plt.xlabel('Age'); plt.ylabel('Frequency')
plt.hist(dataset['km'], color = 'skyblue');plt.title('Histogram of KM'); plt.xlabel('KM'); plt.ylabel('Frequency')
plt.hist(dataset['hp'], color= 'orange');plt.title('Histogram of HP'); plt.xlabel('HP'); plt.ylabel('Frequency')
plt.hist(dataset['cc'], color= 'brown');plt.title('Histogram of CC'); plt.xlabel('CC'); plt.ylabel('Frequency')
plt.hist(dataset['doors'], color = 'violet');plt.title('Histogram of Doors'); plt.xlabel('Doors'); plt.ylabel('Frequency')
plt.hist(dataset['gears'], color= 'lightgreen');plt.title('Histogram of Gears'); plt.xlabel('Gears'); plt.ylabel('Frequency')
plt.hist(dataset['quarterly_tax'], color= 'red');plt.title('Histogram of Quarterly_tax'); plt.xlabel('Quarterly_tax'); plt.ylabel('Frequency')
plt.hist(dataset['weight'], color = 'blue');plt.title('Histogram of Weight'); plt.xlabel('Weight'); plt.ylabel('Frequency')

# Normal Q-Q plot
plt.plot(dataset, alpha=1); plt.legend(['price', 'age', 'km', 'hp', 'cc', 'doors', 'gears', 'quarterly_tax', 'weight'])

price = np.array(dataset['price'])
age = np.array(dataset['age'])
km = np.array(dataset['km'])
hp = np.array(dataset['hp'])
cc = np.array(dataset['cc'])
doors = np.array(dataset['doors'])
gears = np.array(dataset['gears'])
tax = np.array(dataset['quarterly_tax'])
weight = np.array(dataset['weight'])

from scipy import stats
stats.probplot(price, dist='norm', plot=plt);plt.title('Q-Q plot of Price')
stats.probplot(age, dist='norm', plot=plt);plt.title('Q-Q plot of Age')
stats.probplot(km, dist='norm', plot=plt);plt.title('Q-Q plot of KM')
stats.probplot(hp, dist='norm', plot=plt);plt.title('Q-Q plot of HP')
stats.probplot(cc, dist='norm', plot=plt);plt.title('Q-Q plot of CC')
stats.probplot(doors, dist='norm', plot=plt);plt.title('Q-Q plot of Doors')
stats.probplot(gears, dist='norm', plot=plt);plt.title('Q-Q plot of Gears')
stats.probplot(tax, dist='norm', plot=plt);plt.title('Q-Q plot of Quarterly_tax')
stats.probplot(weight, dist='norm', plot=plt);plt.title('Q-Q plot of Weight')

# Normal Probability distribution 
# Price
x_price = np.linspace(np.min(price), np.max(price))
y_price = stats.norm.pdf(x_price, np.mean(x_price), np.std(x_price))
plt.plot(x_price, y_price,); plt.xlim(np.min(x_price), np.max(x_price));plt.xlabel('Price');plt.ylabel('Probability');plt.title('Normal Probability Distribution of Price')

# Age
x_age = np.linspace(np.min(age), np.max(age))
y_age = stats.norm.pdf(x_age, np.mean(x_age), np.std(x_age))
plt.plot(x_age, y_age, color = 'coral'); plt.xlim(np.min(x_age), np.max(x_age));plt.xlabel('Age');plt.ylabel('Probability');plt.title('Normal Probability Distribution of Age')

# KM
x_km = np.linspace(np.min(km), np.max(km))
y_km = stats.norm.pdf(x_km, np.mean(x_km), np.std(x_km))
plt.plot(x_km, y_km, color = 'skyblue'); plt.xlim(np.min(x_km), np.max(x_km));plt.xlabel('KM');plt.ylabel('Probability');plt.title('Normal Probability Distribution of KM')

# HP
x_hp = np.linspace(np.min(hp), np.max(hp))
y_hp = stats.norm.pdf(x_hp, np.mean(x_hp), np.std(x_hp))
plt.plot(x_hp, y_hp, color = 'orange'); plt.xlim(np.min(x_hp), np.max(x_hp));plt.xlabel('HP');plt.ylabel('Probability');plt.title('Normal Probability Distribution of HP')

# CC
x_cc = np.linspace(np.min(cc), np.max(cc))
y_cc = stats.norm.pdf(x_cc, np.mean(x_cc), np.std(x_cc))
plt.plot(x_cc, y_cc, color = 'brown'); plt.xlim(np.min(x_cc), np.max(x_cc));plt.xlabel('CC');plt.ylabel('Probability');plt.title('Normal Probability Distribution of CC')

# Doors
x_doors = np.linspace(np.min(doors), np.max(doors))
y_doors = stats.norm.pdf(x_doors, np.mean(x_doors), np.std(x_doors))
plt.plot(x_doors, y_doors, color = 'violet'); plt.xlim(np.min(x_doors), np.max(x_doors));plt.xlabel('Doors');plt.ylabel('Probability');plt.title('Normal Probability Distribution of Doors')

# Gears
x_gears = np.linspace(np.min(gears), np.max(gears))
y_gears = stats.norm.pdf(x_gears, np.mean(x_gears), np.std(x_gears))
plt.plot(x_gears, y_gears, color = 'lightgreen'); plt.xlim(np.min(x_gears), np.max(x_gears));plt.xlabel('Gears');plt.ylabel('Probability');plt.title('Normal Probability Distribution of Gears')

# Tax
x_tax = np.linspace(np.min(tax), np.max(tax))
y_tax = stats.norm.pdf(x_tax, np.mean(x_tax), np.std(x_tax))
plt.plot(x_tax, y_tax, color = 'red'); plt.xlim(np.min(x_tax), np.max(x_tax));plt.xlabel('Quarterly_tax');plt.ylabel('Probability');plt.title('Normal Probability Distribution of Quarterly_tax')

# Weight
x_weight = np.linspace(np.min(weight), np.max(weight))
y_weight = stats.norm.pdf(x_weight, np.mean(x_weight), np.std(x_weight))
plt.plot(x_weight, y_weight, color = 'blue'); plt.xlim(np.min(x_weight), np.max(x_weight));plt.xlabel('Weight');plt.ylabel('Probability');plt.title('Normal Probability Distribution of Weight')


# Boxplot
import seaborn as sns
sns.boxplot(data=dataset).set_title('Boxplot of Independent Variables')
sns.boxplot(data=dataset['price']).set_title('Boxplot of Price')
sns.boxplot(data=dataset['age'], color='coral').set_title('Boxplot of Age')
sns.boxplot(data=dataset['km'], color='skyblue').set_title('Boxplot of KM')
sns.boxplot(data=dataset['hp'], color='orange').set_title('Boxplot of HP')
sns.boxplot(data=dataset['cc'], color='brown').set_title('Boxplot of CC')
sns.boxplot(data=dataset['doors'], color='violet').set_title('Boxplot of Doors')
sns.boxplot(data=dataset['gears'], color='lightgreen').set_title('Boxplot of Gears')
sns.boxplot(data=dataset['quarterly_tax'], color='lightgreen').set_title('Boxplot of Quarterly_tax')
sns.boxplot(data=dataset['weight'], color = 'lightblue').set_title('Boxplot of Weight')

sns.distplot(dataset['price'], fit=norm, kde=False)
sns.distplot(dataset['age'], fit=norm, kde=False, color = 'coral')
sns.distplot(dataset['km'], fit=norm, kde=False, color = 'skyblue')
sns.distplot(dataset['hp'], fit=norm, kde=False, color = 'orange')
sns.distplot(dataset['cc'], fit=norm, kde=False, color = 'brown')
sns.distplot(dataset['doors'], fit=norm, kde=False, color = 'violet')
sns.distplot(dataset['gears'], fit=norm, kde=False, color = 'red')
sns.distplot(dataset['quarterly_tax'], fit=norm, kde=False, color = 'blue')
sns.distplot(dataset['weight'], fit=norm, kde=False, color = 'lightgreen')

# Scatter plot
sns.scatterplot(x='price', y='age', data=dataset).set_title('Scatterplot of Price & Age')
sns.scatterplot(x='price', y='km', data = dataset).set_title('Scatterplot of Price & KM')
sns.scatterplot(x='price', y='hp', data=dataset).set_title('Scatterplot of Price & HP')
sns.scatterplot(x='price', y='cc', data=dataset).set_title('Scatterplot of Price & CC')
sns.scatterplot(x='price', y='quarterly_tax', data=dataset).set_title('Scatterplot of Price & Quarterly Tax')
sns.scatterplot(x='price', y='weight', data=dataset).set_title('Scatterplot of Price & Weight')
sns.scatterplot(x='age', y='km', data=dataset).set_title('Scatterplot of Age & KM')
sns.scatterplot(x='km', y='weight', data=dataset).set_title('Scatterplot of Price & CC')

sns.pairplot(dataset)
sns.pairplot(dataset, diag_kind = 'kde')
sns.pairplot(dataset, hue = 'hp')
sns.pairplot(dataset, kind = 'reg')

# Heatmap
dataset.corr().round(2)
sns.heatmap(dataset.corr(), annot=True)

# Detecting and removing outliers
# Z-score
from scipy import stats
z = np.abs(stats.zscore(dataset))
print(z)
threshold = 3
print(np.where(z>3))
print(z[1][7])

df_out = dataset[(z<3).all(axis=1)] # 120 rows containing outliers
df_out.shape

# Metrics of features
X = df_out.iloc[:, 1:].values
Y = df_out.iloc[:, 0].values

# Splitting the data into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Fitting the MultipleLinearRegression on train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize = True)
regressor.fit(X_train, Y_train)

# Predicting train set results
reg_train_pred = regressor.predict(X_train)
# Train set residuals
reg_train_resid = reg_train_pred - Y_train
# RMSE
reg_train_rmse = np.sqrt(np.mean(reg_train_resid**2))
print(reg_train_rmse) # 1136.814

# Predicting test set results
reg_test_pred = regressor.predict(X_test)
# test set residuals
reg_test_resid = reg_test_pred - Y_test
# RMSE
reg_test_rmse = np.sqrt(np.mean(reg_test_resid**2))
print(reg_test_rmse) # 1176.621

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as smf
X = np.append(arr=np.ones((1315,1)).astype(int), values=X, axis=1) # Adding 1's matrix to X

# With all the independent variables
X_opt = X[:, [0,1,2,3,4,5,6,7,8]]
model1 = smf.OLS(endog=Y, exog=X_opt).fit()
model1.summary() # R^2 = 0.858 & Adj.R^2 = 0.857 
model1.params
# Confidence value 99%
print(model1.conf_int(0.01))

# Removing X7
X_opt = X[:, [0,1,2,3,4,5,6,8]]
model2 = smf.OLS(endog=Y, exog=X_opt).fit()
model2.summary() #  R^2 = 0.858 & Adj.R^2 = 0.857 
model2.params
# Confidence value 99%
print(model2.conf_int(0.01)) # All are significant

# Influence index plot
import statsmodels.api as sm
sm.graphics.influence_plot(model1) # Index 138, 137, 338,330, 938, 135, 457, 1314  has highest influence value
# studentized residuals = residuals/ std.deviation of residuals

new_X = np.delete(X, [135,137,138,330,338,457,938,1314], axis=0)
new_X_opt = new_X[:, [0,1,2,3,4,5,6,7,8]]
new_Y = np.delete(Y, [135,137,138,330,338,457,938,1314], axis=0)

# Preparing new model
model1_new = smf.OLS(endog = new_Y, exog=new_X_opt).fit()
model1_new.summary() # R^2 = 0.871 & Adj.R^2 = 0.870
model1_new.params
# confidence value 99%
print(model1_new.conf_int(0.01))

# remove X7
new_X_opt = new_X[:, [0,1,2,3,4,5,6,8]]
model2_new = smf.OLS(endog=new_Y, exog=new_X_opt).fit()
model2_new.summary() # R^2 = 0.871 & Adj.R^2 = 0.870 (same as above) so don't remove the X7 
model2_new.params
# confidence value 99%
print(model2_new.conf_int(0.01))

# Calculating VIF's Values of independent variables
rsq_age = smf.OLS(endog=X[:,1], exog=X[:, [0,2,3,4,5,6,7,8]]).fit().rsquared
vif_age = 1/(1-rsq_age) # 1.95

rsq_km = smf.OLS(endog=X[:,2], exog=X[:, [0,1,3,4,5,6,7,8]]).fit().rsquared
vif_km = 1/(1-rsq_km) # 1.83

rsq_hp = smf.OLS(endog=X[:,3], exog=X[:, [0,1,2,4,5,6,7,8]]).fit().rsquared
vif_hp = 1/(1-rsq_hp) # 1.59

rsq_cc = smf.OLS(endog=X[:,4], exog=X[:, [0,1,2,3,5,6,7,8]]).fit().rsquared
vif_cc = 1/(1-rsq_cc) # 3.00

rsq_doors = smf.OLS(endog=X[:,5], exog=X[:, [0,1,2,3,4,6,7,8]]).fit().rsquared
vif_doors = 1/(1-rsq_doors) # 1.38

rsq_gears = smf.OLS(endog=X[:,6], exog=X[:, [0,1,2,3,4,5,7,8]]).fit().rsquared
vif_gears = 1/(1-rsq_gears) # 0

rsq_tax = smf.OLS(endog=X[:,7], exog=X[:, [0,1,2,3,4,5,6,8]]).fit().rsquared
vif_tax = 1/(1-rsq_tax) # 2.35

rsq_weight = smf.OLS(endog=X[:,8], exog=X[:, [0,1,2,3,4,5,6,7]]).fit().rsquared
vif_weight = 1/(1-rsq_weight) # 3.35

# Sorting VIF's values in dataframe
d1 = {'variables':['age', 'km', 'hp', 'cc', 'doors', 'gears', 'quarterly_tax', 'weight'], 'VIF':[vif_age, vif_km, vif_hp, vif_cc, vif_doors, vif_gears, vif_tax, vif_weight]}
vif_frame = pd.DataFrame(d1)
vif_frame # All variables satisfies the VIF < 10

# Added Variable Plot
sm.graphics.plot_partregress_grid(model1_new)
# X7 looks slightly varing 
# X6 variable is not satisfies but after removing the X6 R^2 value not changed so i don't remove that.
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
print(train_rmse) # 1093.51 

# prediction on test set data
test_pred = final_model.predict(newX_test)

# Test set residual values
test_resid = test_pred - newY_test

# RMSE value for test data
test_rmse = np.sqrt(np.mean(test_resid**2))
print(test_rmse) # 1054.65 
