# Reset the console
%reset -f

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset
dataset = pd.read_csv('Company_Data.csv')
dataset.head()
dataset.columns
dataset.info()
dataset.dtypes
dataset.isnull().sum() # No missing values
dataset.shape
dataset.drop_duplicates(keep='first',inplace=True) # No duplicate rows

# numerical data
num_data = dataset.select_dtypes(include=['float64','int64'])
cat_data = dataset.select_dtypes(include='object')

# Statistical Description
dataset.describe()
# Measure of Dispersion
np.var(dataset)
np.std(dataset)
# Skewness and Kurtosis
from scipy.stats import skew, kurtosis
skew(num_data)
kurtosis(num_data)

# Histogram
plt.hist(num_data['Sales']);plt.title('Histogram of Sales');plt.xlabel('No.of Sales');plt.ylabel('Frequency')
plt.hist(num_data['CompPrice'], color='coral');plt.title('Histogram of Competitor Price');plt.xlabel('Competitor Price');plt.ylabel('Frequency')
plt.hist(num_data['Income'], color='skyblue');plt.title('Histogram of Income');plt.xlabel('Income');plt.ylabel('Frequency')
plt.hist(num_data['Advertising'], color='orange');plt.title('Histogram of Advertising');plt.xlabel('Advertising');plt.ylabel('Frequency')
plt.hist(num_data['Population'], color='teal');plt.title('Histogram of Population');plt.xlabel('Population');plt.ylabel('Frequency')
plt.hist(num_data['Price'], color='purple');plt.title('Histogram of Price');plt.xlabel('Prices');plt.ylabel('Frequency')
plt.hist(num_data['Age'], color='brown');plt.title('Histogram of Age');plt.xlabel('Age');plt.ylabel('Frequency')
plt.hist(num_data['Education'], color='violet');plt.title('Histogram of Education');plt.xlabel('Education');plt.ylabel('Frequency')

# Coutplot for categorical data
import seaborn as sns
sns.countplot(cat_data['ShelveLoc']).set_title('Count of ShelveLoc\'s')
sns.countplot(cat_data['Urban']).set_title('Count of Urban\'s')
sns.countplot(cat_data['US']).set_title('Count of US Residences')

# Normal Q-Q plot
plt.plot(num_data);plt.legend(list(num_data.columns))

sales = np.array(num_data['Sales'])
comprice = np.array(num_data['CompPrice'])
income= np.array(num_data['Income'])
advt = np.array(num_data['Advertising'])
popl = np.array(num_data['Population'])
price = np.array(num_data['Price'])
age = np.array(num_data['Age'])
educt = np.array(num_data['Education'])

from scipy import stats
stats.probplot(sales, dist='norm', plot=plt);plt.title('Probability plot of Sales')
stats.probplot(comprice, dist='norm', plot=plt);plt.title('Probability plot of Competitor Price')
stats.probplot(income, dist='norm', plot=plt);plt.title('Probability plot of Income')
stats.probplot(advt, dist='norm', plot=plt);plt.title('Probability plot of Advertising')
stats.probplot(popl, dist='norm', plot=plt);plt.title('Probability plot of Population')
stats.probplot(price, dist='norm', plot=plt);plt.title('Probability plot of Price')
stats.probplot(age, dist='norm', plot=plt);plt.title('Probability plot of Age')
stats.probplot(educt, dist='norm', plot=plt);plt.title('Probability plot of Education')

# Normal Probabiliry Distribution
x_sales = np.linspace(np.min(sales), np.max(sales))
y_sales = stats.norm.pdf(x_sales, np.median(x_sales), np.std(x_sales))
plt.plot(x_sales, y_sales);plt.xlim(np.min(sales), np.max(sales));plt.title('Normal Probability Distribution of Sales');plt.xlabel('No.of sales');plt.ylabel('Probability')

x_comprice = np.linspace(np.min(comprice), np.max(comprice))
y_comprice = stats.norm.pdf(x_comprice, np.median(x_comprice), np.std(x_comprice))
plt.plot(x_comprice, y_comprice);plt.xlim(np.min(comprice), np.max(comprice));plt.title('Normal Probability Distribution of Competitor Price');plt.xlabel('Competitor Price');plt.ylabel('Probability')

x_income = np.linspace(np.min(income), np.max(income))
y_income = stats.norm.pdf(x_income, np.median(x_income), np.std(x_income))
plt.plot(x_income, y_income);plt.xlim(np.min(income), np.max(income));plt.title('Normal Probability Distribution of Income');plt.xlabel('Income');plt.ylabel('Probability')

x_advt = np.linspace(np.min(advt), np.max(advt))
y_advt = stats.norm.pdf(x_advt, np.median(x_advt), np.std(x_advt))
plt.plot(x_advt, y_advt);plt.xlim(np.min(advt), np.max(advt));plt.title('Normal Probability Distribution of Advertising');plt.xlabel('Advertising ');plt.ylabel('Probability')

x_popl = np.linspace(np.min(popl), np.max(popl))
y_popl = stats.norm.pdf(x_popl, np.median(x_popl), np.std(x_popl))
plt.plot(x_popl, y_popl);plt.xlim(np.min(popl), np.max(popl));plt.title('Normal Probability Distribution of Population');plt.xlabel('Population');plt.ylabel('Probability')

x_price = np.linspace(np.min(price), np.max(price))
y_price = stats.norm.pdf(x_price, np.median(x_price), np.std(x_price))
plt.plot(x_price, y_price);plt.xlim(np.min(price), np.max(price));plt.title('Normal Probability Distribution of Price');plt.xlabel('Price');plt.ylabel('Probability')

x_age = np.linspace(np.min(age), np.max(age))
y_age = stats.norm.pdf(x_age, np.median(x_age), np.std(x_age))
plt.plot(x_age, y_age);plt.xlim(np.min(age), np.max(age));plt.title('Normal Probability Distribution of Age');plt.xlabel('Age');plt.ylabel('Probability')

x_educt = np.linspace(np.min(educt), np.max(educt))
y_educt = stats.norm.pdf(x_educt, np.median(x_educt), np.std(x_educt))
plt.plot(x_educt, y_educt);plt.xlim(np.min(educt), np.max(educt));plt.title('Normal Probability Distribution of Education');plt.xlabel('Education');plt.ylabel('Probability')

# Boxplot
sns.boxplot(num_data['Sales'],orient='v').set_title('Boxplot of Sales')
sns.boxplot(num_data['CompPrice'],orient='v', color='coral').set_title('Boxplot of Competitor Price')
sns.boxplot(num_data['Income'],orient='v', color='skyblue').set_title('Boxplot of Income')
sns.boxplot(num_data['Advertising'],orient='v', color='orange').set_title('Boxplot of Advertising')
sns.boxplot(num_data['Population'],orient='v', color='teal').set_title('Boxplot of Population')
sns.boxplot(num_data['Price'],orient='v', color='purple').set_title('Boxplot of Price')
sns.boxplot(num_data['Age'],orient='v', color='brown').set_title('Boxplot of Age')
sns.boxplot(num_data['Education'],orient='v', color='violet').set_title('Boxplot of Education')

# Boxplot of numerical data wrt categorical data
sns.boxplot(x='ShelveLoc', y='Sales', data=dataset).set_title('Boxplot of Shelveloc & Sales')
sns.boxplot(x='ShelveLoc', y='CompPrice', data=dataset).set_title('Boxplot of Shelveloc & CompPrice')
sns.boxplot(x='ShelveLoc', y='Income', data=dataset).set_title('Boxplot of Shelveloc & Income')
sns.boxplot(x='ShelveLoc', y='Advertising', data=dataset).set_title('Boxplot of Shelveloc & Advertising')
sns.boxplot(x='ShelveLoc', y='Population', data=dataset).set_title('Boxplot of Shelveloc & Population')
sns.boxplot(x='ShelveLoc', y='Price', data=dataset).set_title('Boxplot of Shelveloc & Price')
sns.boxplot(x='ShelveLoc', y='Age', data=dataset).set_title('Boxplot of Shelveloc & Age')
sns.boxplot(x='ShelveLoc', y='Education', data=dataset).set_title('Boxplot of Shelveloc & Education')

sns.boxplot(x='Urban', y='Sales', data=dataset).set_title('Boxplot of Urban & Sales')
sns.boxplot(x='Urban', y='CompPrice', data=dataset).set_title('Boxplot of Urban & CompPrice')
sns.boxplot(x='Urban', y='Income', data=dataset).set_title('Boxplot of Urban & Income')
sns.boxplot(x='Urban', y='Advertising', data=dataset).set_title('Boxplot of Urban & Advertising')
sns.boxplot(x='Urban', y='Population', data=dataset).set_title('Boxplot of Urban & Population')
sns.boxplot(x='Urban', y='Price', data=dataset).set_title('Boxplot of Urban & Price')
sns.boxplot(x='Urban', y='Age', data=dataset).set_title('Boxplot of Urban & Age')
sns.boxplot(x='Urban', y='Education', data=dataset).set_title('Boxplot of Urban & Education')

sns.boxplot(x='US', y='Sales', data=dataset).set_title('Boxplot of US & Sales')
sns.boxplot(x='US', y='CompPrice', data=dataset).set_title('Boxplot of US & CompPrice')
sns.boxplot(x='US', y='Income', data=dataset).set_title('Boxplot of US & Income')
sns.boxplot(x='US', y='Advertising', data=dataset).set_title('Boxplot of US & Advertising')
sns.boxplot(x='US', y='Population', data=dataset).set_title('Boxplot of US & Population')
sns.boxplot(x='US', y='Price', data=dataset).set_title('Boxplot of US & Price')
sns.boxplot(x='US', y='Age', data=dataset).set_title('Boxplot of US & Age')
sns.boxplot(x='US', y='Education', data=dataset).set_title('Boxplot of US & Education')

# Scatterplot
sns.scatterplot(x='Sales', y='CompPrice', data=dataset).set_title('Scatterplot of Sales & Competitor Price')
sns.scatterplot(x='Sales', y='Income', data=dataset, color='coral').set_title('Scatterplot of Sales & Income')
sns.scatterplot(x='Sales', y='Advertising', data=dataset).set_title('Scatterplot of Sales & Advertising')
sns.scatterplot(x='Sales', y='Population', data=dataset).set_title('Scatterplot of Sales & Population')
sns.scatterplot(x='Sales', y='Price', data=dataset).set_title('Scatterplot of Sales & Price')
sns.scatterplot(x='Sales', y='Age', data=dataset).set_title('Scatterplot of Sales & Age')
sns.scatterplot(x='Sales', y='Education', data=dataset).set_title('Scatterplot of Sales & Education')

sns.scatterplot(x='CompPrice', y='Income', data=dataset, color='coral').set_title('Scatterplot of Competitor Price & Income')
sns.scatterplot(x='CompPrice', y='Advertising', data=dataset).set_title('Scatterplot of Competitor Price & Advertising')
sns.scatterplot(x='CompPrice', y='Population', data=dataset).set_title('Scatterplot of Competitor Price & Population')
sns.scatterplot(x='CompPrice', y='Price', data=dataset).set_title('Scatterplot of Competitor Price & Price')
sns.scatterplot(x='CompPrice', y='Age', data=dataset).set_title('Scatterplot of Competitor Price & Age')
sns.scatterplot(x='CompPrice', y='Education', data=dataset).set_title('Scatterplot of Competitor Price & Education')

sns.scatterplot(x='Income', y='Advertising', data=dataset).set_title('Scatterplot of Income & Advertising')
sns.scatterplot(x='Income', y='Population', data=dataset).set_title('Scatterplot of Income & Population')
sns.scatterplot(x='Income', y='Price', data=dataset).set_title('Scatterplot of Income & Price')
sns.scatterplot(x='Income', y='Age', data=dataset).set_title('Scatterplot of Income & Age')
sns.scatterplot(x='Income', y='Education', data=dataset).set_title('Scatterplot of Income & Education')

sns.scatterplot(x='Advertising', y='Population', data=dataset).set_title('Scatterplot of Income & Population')
sns.scatterplot(x='Advertising', y='Price', data=dataset).set_title('Scatterplot of Income & Price')
sns.scatterplot(x='Advertising', y='Age', data=dataset).set_title('Scatterplot of Income & Age')
sns.scatterplot(x='Advertising', y='Education', data=dataset).set_title('Scatterplot of Income & Education')

sns.scatterplot(x='Population', y='Price', data=dataset).set_title('Scatterplot of Population & Price')
sns.scatterplot(x='Population', y='Age', data=dataset).set_title('Scatterplot of Population & Age')
sns.scatterplot(x='Population', y='Education', data=dataset).set_title('Scatterplot of Population & Education')

sns.scatterplot(x='Price', y='Age', data=dataset).set_title('Scatterplot of Price & Age')
sns.scatterplot(x='Price', y='Education', data=dataset).set_title('Scatterplot of Price & Education')

sns.scatterplot(x='Age', y='Education', data=dataset).set_title('Scatterplot of Population & Education')

sns.pairplot(dataset)
sns.pairplot(dataset, diag_kind='kde')
sns.pairplot(dataset, hue='US')
sns.pairplot(dataset, hue='Urban')

# Heatmap
corr = dataset.corr()
sns.heatmap(corr, annot=True)

# Creating dummy variables
final_data = pd.get_dummies(dataset)
# Outliers
from scipy import stats
Z = np.abs(stats.zscore(final_data))
print(np.where(Z>3))
print(Z[42][5])
final_data = final_data[(Z<3).all(axis=1)] # 4 outliers are removed
final_data.shape

colnames = list(final_data.columns)
predictors = colnames[1:]
target = colnames[0]

# Metrics of features
X = final_data[predictors].values
Y = final_data[target].values

# Splitting Data into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state =0)

# Fitting Decision Tree onto Training set
from sklearn.tree import DecisionTreeRegressor
classifier = DecisionTreeRegressor(criterion ='mse', random_state =0)
classifier.fit(X_train, Y_train)

# predicting train set results
from sklearn.metrics import confusion_matrix, accuracy_score
train_pred = classifier.predict(X_train)
# train residual values
train_resid = train_pred - Y_train
# RMSE value of train data
train_rmse = np.sqrt(np.mean(train_resid**2))
print(train_rmse) # 0

# Predicting test set results
test_pred = classifier.predict(X_test)
classifier.score(Y_test, test_pred)
classifier.get_n_leaves() # 295 leaves

# Test set residual values
test_resid = test_pred - Y_test

# RMSE value for test data
test_rmse = np.sqrt(np.mean(test_resid**2))
print(test_rmse) # 2.3780

# Visualizing the Tree
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

# Fitting Decision Tree onto Training set
from sklearn.tree import DecisionTreeRegressor
classifier = DecisionTreeRegressor(criterion ='mae', random_state =0)
classifier.fit(X_train, Y_train)

# predicting train set results
from sklearn.metrics import confusion_matrix, accuracy_score
train_pred = classifier.predict(X_train)
# train residual values
train_resid = train_pred - Y_train
# RMSE value of train data
train_rmse = np.sqrt(np.mean(train_resid**2))
print(train_rmse) # 0

# Predicting test set results
test_pred = classifier.predict(X_test)
classifier.score(Y_test, test_pred)
classifier.get_n_leaves() # 292 leaves

# Test set residual values
test_resid = test_pred - Y_test

# RMSE value for test data
test_rmse = np.sqrt(np.mean(test_resid**2))
print(test_rmse) # 2.3191
