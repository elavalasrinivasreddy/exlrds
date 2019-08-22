# Reset the console
%reset -f

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
dataset = pd.read_csv('50_Startups.csv')
dataset.head()
dataset.dtypes

dataset.info()
dataset.isnull().sum() # No missing values
dataset.shape
dataset.drop_duplicates(keep='first', inplace=True) # No duplicates
dataset.columns

# Statistical Description
dataset.describe()
# Measures of Dispersion
np.var(dataset)
np.std(dataset)
# Skewness and Kurtosis
from scipy.stats import skew, kurtosis
skew(dataset.drop('State', axis=1))
kurtosis(dataset.drop('State', axis=1))

# Histogram 
plt.hist(dataset['R&D Spend']);plt.title('Histogram of R&D Spend');plt.xlabel('R&D Spend');plt.ylabel('Frequency')
plt.hist(dataset['Administration'], color='coral');plt.title('Histogram of Administration');plt.xlabel('Administration');plt.ylabel('Frequency')
plt.hist(dataset['Marketing Spend'], color='teal');plt.title('Histogram of Marketing Spend');plt.xlabel('Marketing Spend');plt.ylabel('Frequency')
plt.hist(dataset['Profit'], color='purple');plt.title('Histogram of Profit');plt.xlabel('Profit');plt.ylabel('Frequency')

# Barplot
import seaborn as sns
sns.countplot(dataset['State']).set_title('Count of State')

# Normal Q-Q plot
plt.plot(dataset.drop('State', axis=1));plt.legend(['R&D Spend', 'Administration', 'Marketing Spend', 'Profit'])

rd = np.array(dataset['R&D Spend'])
admin = np.array(dataset['Administration'])
mark = np.array(dataset['Marketing Spend'])
profit = np.array(dataset['Profit'])

from scipy import stats
stats.probplot(rd, dist='norm', plot=plt);plt.title('Probability Plot of R&D Spend')
stats.probplot(admin, dist='norm', plot=plt);plt.title('Probability Plot of Administration')
stats.probplot(mark, dist='norm',plot=plt);plt.title('Probability Plot of Marketing Spend')
stats.probplot(profit, dist='norm', plot=plt);plt.title('Probability Plot of Profit')

# Normal Probability Distribution
x_rd = np.linspace(np.min(rd), np.max(rd))
y_rd = stats.norm.pdf(x_rd, np.median(x_rd), np.std(x_rd))
plt.plot(x_rd, y_rd);plt.xlim(np.min(rd), np.max(rd));plt.title('Normal Probability Distribution of R&D Spend');plt.xlabel('R&D Spend');plt.ylabel('Probability')

x_admin = np.linspace(np.min(admin), np.max(admin))
y_admin = stats.norm.pdf(x_admin, np.median(x_admin), np.std(x_admin))
plt.plot(x_admin, y_admin);plt.xlim(np.min(admin), np.max(admin));plt.title('Normal Probability Distribution of Administration');plt.xlabel('Administration');plt.ylabel('Probability')

x_mark = np.linspace(np.min(mark), np.max(mark))
y_mark = stats.norm.pdf(x_mark, np.median(x_mark), np.std(x_mark))
plt.plot(x_mark, y_mark);plt.xlim(np.min(mark), np.max(mark));plt.title('Normal Probability Distribution of Marketing Spend');plt.xlabel('Marketing Spend');plt.ylabel('Probability')

x_profit = np.linspace(np.min(profit), np.max(profit))
y_profit = stats.norm.pdf(x_profit, np.median(x_profit), np.std(x_profit))
plt.plot(x_profit, y_profit);plt.xlim(np.min(profit), np.max(profit));plt.title('Normal Probability Distribution of Profit');plt.xlabel('Profit');plt.ylabel('Probability')

# Boxplot
sns.boxplot(dataset['R&D Spend'], orient='v').set_title('Boxplot of R&D Spend')
sns.boxplot(dataset['Administration'], orient='v', color='coral').set_title('Boxplot of Administration')
sns.boxplot(dataset['Marketing Spend'], orient='v', color='teal').set_title('Boxplot of Marketing Spend')
sns.boxplot(dataset['Profit'], orient='v', color='purple').set_title('Boxplot of Profit')

# Boxplot wrt State
sns.boxplot(x='State', y='R&D Spend', data=dataset).set_title('Boxplot of State & R&D Spend')
sns.boxplot(x='State', y='Administration', data=dataset).set_title('Boxplot of State & Administration')
sns.boxplot(x='State', y='Marketing Spend', data=dataset).set_title('Boxplot of State & Marketing Spend')
sns.boxplot(x='State', y='Profit', data=dataset).set_title('Boxplot of State & Profit')

# Scatterplot
sns.scatterplot(x='R&D Spend', y='Administration', data=dataset).set_title('Scatterplot of R&D Spend & Administration')
sns.scatterplot(x='R&D Spend', y='Marketing Spend', data=dataset).set_title('Scatterplot of R&D Spend & Marketing Spend')
sns.scatterplot(x='R&D Spend', y='Profit', data=dataset).set_title('Scatterplot of R&D Spend & Profit')

sns.scatterplot(x='Administration', y='Marketing Spend', data=dataset).set_title('Scatterplot of Administration & Marketing Spend')
sns.scatterplot(x='Administration', y='Profit', data=dataset).set_title('Scatterplot of Administration & Profit')

sns.scatterplot(x='Marketing Spend', y='Profit', data=dataset).set_title('Scatterplot of Marketing Spend & Profit')

sns.pairplot(dataset)
sns.pairplot(dataset, hue='State', diag_kind='kde')

# Heatmap
corr = dataset.corr()
sns.heatmap(corr, annot = True)

# Outliers
from scipy.stats import zscore
Z = np.abs(zscore(dataset.drop('State', axis=1)))
print(np.where(Z>3)) # No outliers

# Creating Dummy variables for State
dataset = pd.get_dummies(dataset)

# Splitting the data into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dataset.drop('Profit', axis=1), dataset['Profit'], test_size=0.30, random_state=0)

# Normalization
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# Build the ANN Model
from keras.models import Sequential
from keras.layers import Dense

regressor = Sequential()
# Adding the input layer and hidden layer
regressor.add(Dense(30, input_dim=6, kernel_initializer='normal', activation='relu'))
# Adding 2nd hidden layer
regressor.add(Dense(24, activation='relu'))
# Adding 3rd hidden layer
regressor.add(Dense(12, activation='relu'))
# Adding 4th hidden layer
regressor.add(Dense(6, activation='relu'))
# Adding Output layer
regressor.add(Dense(1, activation='relu'))
# Compile the model
regressor.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Fitting the ANN model on data
regressor.fit((X_train), (Y_train), epochs=600, batch_size=600)
# [600,600],[600,300],[1500,1500],[1500,1000],[1500,150]

# Predicting the results on train set
train_pred = regressor.predict(X_train)
train_pred = pd.Series([i[0] for i in train_pred])
train_error = Y_train - train_pred
train_rmse = np.sqrt(np.mean(train_error**2))
print(train_rmse) # 53095.9662

plt.plot(train_pred, Y_train,"ro")
np.corrcoef(train_pred, Y_train) # 0.99

# Predicting the results on test set
test_pred = regressor.predict(X_test)
test_pred = pd.Series([i[0] for i in test_pred])
test_error = Y_test - test_pred
test_rmse = np.sqrt(np.mean(test_error**2))
print(test_rmse) # 45060.9456

plt.plot(test_pred,Y_test,"bo")
np.corrcoef(test_pred, Y_test) # 0.95

# Visualize the ANN model
from ann_visualizer.visualize import ann_viz
ann_viz(regressor, title='NeuralNetwork for 50_Startups')

# Another model from sklearn
from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(hidden_layer_sizes=(6,468))
regressor.fit(X_train, Y_train)

# Predicting the results on train set
pred_train = regressor.predict(X_train)
train_error = Y_train - pred_train
train_rmse = np.sqrt(np.mean(train_error**2))
print(train_rmse) # 118106.0476

# Predicting the results on test set
pred_test = regressor.predict(X_test)
test_error = Y_test - pred_test
test_rmse = np.sqrt(np.mean(test_error**2))
print(test_rmse) # 119972.6471
