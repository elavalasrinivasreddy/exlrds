#Reset the console
%reset -f

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('concrete.csv')
dataset.head()
dataset.columns
dataset.dtypes

dataset.shape
dataset.isnull().sum() # No missing Values
dataset.drop_duplicates(keep='first', inplace=True) # 25 duplicate rows there

dataset.info()
# Statistical Description
dataset.describe()
# Measure of Dispersion
np.var(dataset)
np.std(dataset)
# Skewness and Kurtosis
from scipy.stats import skew, kurtosis
skew(dataset)
kurtosis(dataset)

# Histogram 
plt.hist(dataset['cement']);plt.title('Histogram of Cement');plt.xlabel('Cement');plt.ylabel('Frequency')
plt.hist(dataset['slag'], color='coral');plt.title('Histogram of Slag');plt.xlabel('Slag');plt.ylabel('Frequency')
plt.hist(dataset['ash'], color='skyblue');plt.title('Histogram of Ash');plt.xlabel('Ash');plt.ylabel('Frequency')
plt.hist(dataset['water'], color='orange');plt.title('Histogram of Water');plt.xlabel('Water');plt.ylabel('Frequency')
plt.hist(dataset['superplastic'], color='teal');plt.title('Histogram of SuperPlastic');plt.xlabel('SuperPlastic');plt.ylabel('Frequency')
plt.hist(dataset['coarseagg'], color='brown');plt.title('Histogram of Coarseagg');plt.xlabel('Coarseagg');plt.ylabel('Frequency')
plt.hist(dataset['fineagg'], color='violet');plt.title('Histogram of Fineagg');plt.xlabel('Fineagg');plt.ylabel('Frequency')
plt.hist(dataset['age'], color='purple');plt.title('Histogram of Age');plt.xlabel('Age');plt.ylabel('Frequency')
plt.hist(dataset['strength'], color='lightgreen');plt.title('Histogram of Strength');plt.xlabel('Strength');plt.ylabel('Frequency')

# Normal Q-Q plot
plt.plot(dataset);plt.legend(list(dataset.columns))

cement = np.array(dataset['cement'])
slag = np.array(dataset['slag'])
ash = np.array(dataset['ash'])
water = np.array(dataset['water'])
supplst = np.array(dataset['superplastic'])
coaragg = np.array(dataset['coarseagg'])
fineagg = np.array(dataset['fineagg'])
age = np.array(dataset['age'])
strength = np.array(dataset['strength'])

from scipy import stats
stats.probplot(cement, dist='norm', plot=plt);plt.title('Probability Plot of Cement')
stats.probplot(slag, dist='norm', plot=plt);plt.title('Probability Plot of Slag')
stats.probplot(ash, dist='norm', plot=plt);plt.title('Probability Plot of Ash')
stats.probplot(water, dist='norm', plot=plt);plt.title('Probability Plot of Water')
stats.probplot(supplst, dist='norm', plot=plt);plt.title('Probability Plot of Superplastic')
stats.probplot(coaragg, dist='norm', plot=plt);plt.title('Probability Plot of Coarseagg')
stats.probplot(fineagg, dist='norm', plot=plt);plt.title('Probability Plot of Fineagg')
stats.probplot(age, dist='norm', plot=plt);plt.title('Probability Plot of Age')
stats.probplot(strength, dist='norm', plot=plt);plt.title('Probability Plot of Strength')

# Normal Probability Distribution
x_cement = np.linspace(np.min(cement), np.max(cement))
y_cement = stats.norm.pdf(x_cement, np.median(x_cement), np.std(x_cement))
plt.plot(x_cement, y_cement);plt.xlim(np.min(cement), np.max(cement));plt.title('Normal Probability Distribution of Cement');plt.xlabel('Cement');plt.ylabel('Probability')

x_slag = np.linspace(np.min(slag), np.max(slag))
y_slag = stats.norm.pdf(x_slag, np.median(x_slag), np.std(x_slag))
plt.plot(x_slag, y_slag);plt.xlim(np.min(slag), np.max(slag));plt.title('Normal Probability Distribution of Slag');plt.xlabel('Slag');plt.ylabel('Probability')

x_ash = np.linspace(np.min(ash), np.max(ash))
y_ash = stats.norm.pdf(x_ash, np.median(x_ash), np.std(x_ash))
plt.plot(x_ash, y_ash);plt.xlim(np.min(ash), np.max(ash));plt.title('Normal Probability Distribution of Ash');plt.xlabel('Ash');plt.ylabel('Probability')

x_water = np.linspace(np.min(water), np.max(water))
y_water = stats.norm.pdf(x_water, np.median(x_water), np.std(x_water))
plt.plot(x_water, y_water);plt.xlim(np.min(water), np.max(water));plt.title('Normal Probability Distribution of Water');plt.xlabel('Water');plt.ylabel('Probability')

x_supplst = np.linspace(np.min(supplst), np.max(supplst))
y_supplst = stats.norm.pdf(x_supplst, np.median(x_supplst), np.std(x_supplst))
plt.plot(x_supplst, y_supplst);plt.xlim(np.min(supplst), np.max(supplst));plt.title('Normal Probability Distribution of SuperPlastic');plt.xlabel('SuperPlastic');plt.ylabel('Probability')

x_coaragg = np.linspace(np.min(coaragg), np.max(coaragg))
y_coaragg = stats.norm.pdf(x_coaragg, np.median(x_coaragg), np.std(x_coaragg))
plt.plot(x_coaragg, y_coaragg);plt.xlim(np.min(coaragg), np.max(coaragg));plt.title('Normal Probability Distribution of Coarseagg');plt.xlabel('Coarseagg');plt.ylabel('Probability')

x_fineagg = np.linspace(np.min(fineagg), np.max(fineagg))
y_fineagg = stats.norm.pdf(x_fineagg, np.median(x_fineagg), np.std(x_fineagg))
plt.plot(x_fineagg, y_fineagg);plt.xlim(np.min(fineagg), np.max(fineagg));plt.title('Normal Probability Distribution of Fineagg');plt.xlabel('Fineagg');plt.ylabel('Probability')

x_age = np.linspace(np.min(age), np.max(age))
y_age = stats.norm.pdf(x_age, np.median(x_age), np.std(x_age))
plt.plot(x_age, y_age);plt.xlim(np.min(age), np.max(age));plt.title('Normal Probability Distribution of Age');plt.xlabel('Age');plt.ylabel('Probability')

x_strength = np.linspace(np.min(strength), np.max(strength))
y_strength = stats.norm.pdf(x_strength, np.median(x_strength), np.std(x_strength))
plt.plot(x_strength, y_strength);plt.xlim(np.min(strength), np.max(strength));plt.title('Normal Probability Distribution of Strength');plt.xlabel('Strength');plt.ylabel('Probability')

# Boxplot 
import seaborn as sns
sns.boxplot(dataset['cement'], orient='v').set_title('Boxplot of Cement')
sns.boxplot(dataset['slag'], orient='v', color='coral').set_title('Boxplot of Slag')
sns.boxplot(dataset['ash'], orient='v', color='skyblue').set_title('Boxplot of Ash')
sns.boxplot(dataset['water'], orient='v', color='orange').set_title('Boxplot of Water')
sns.boxplot(dataset['superplastic'], orient='v', color='teal').set_title('Boxplot of SuperPlastic')
sns.boxplot(dataset['coarseagg'], orient='v', color='brown').set_title('Boxplot of Coarseagg')
sns.boxplot(dataset['fineagg'], orient='v', color='violet').set_title('Boxplot of Fineagg')
sns.boxplot(dataset['age'], orient='v', color='purple').set_title('Boxplot of Age')
sns.boxplot(dataset['strength'], orient='v', color='lightgreen').set_title('Boxplot of Strength')

# Scatterplot
sns.scatterplot(x='cement', y='slag', data=dataset).set_title('Scatterplot of Cement & Slag')
sns.scatterplot(x='cement', y='ash', data=dataset).set_title('Scatterplot of Cement & Ash')
sns.scatterplot(x='cement', y='water', data=dataset).set_title('Scatterplot of Cement & Water')
sns.scatterplot(x='cement', y='superplastic', data=dataset).set_title('Scatterplot of Cement & SuperPlastic')
sns.scatterplot(x='cement', y='coarseagg', data=dataset).set_title('Scatterplot of Cement & Coarseagg')
sns.scatterplot(x='cement', y='fineagg', data=dataset).set_title('Scatterplot of Cement & Fineagg')
sns.scatterplot(x='cement', y='age', data=dataset).set_title('Scatterplot of Cement & Age')
sns.scatterplot(x='cement', y='strength', data=dataset).set_title('Scatterplot of Cement & Strength')

sns.scatterplot(x='slag', y='ash', data=dataset).set_title('Scatterplot of Slag & Ash')
sns.scatterplot(x='slag', y='water', data=dataset).set_title('Scatterplot of Slag & Water')
sns.scatterplot(x='slag', y='superplastic', data=dataset).set_title('Scatterplot of Slag & SuperPlastic')
sns.scatterplot(x='slag', y='coarseagg', data=dataset).set_title('Scatterplot of Slag & Coarseagg')
sns.scatterplot(x='slag', y='fineagg', data=dataset).set_title('Scatterplot of Slag & Fineagg')
sns.scatterplot(x='slag', y='age', data=dataset).set_title('Scatterplot of Slag & Age')
sns.scatterplot(x='slag', y='strength', data=dataset).set_title('Scatterplot of Slag & Strength')

sns.scatterplot(x='ash', y='water', data=dataset).set_title('Scatterplot of Ash & Water')
sns.scatterplot(x='ash', y='superplastic', data=dataset).set_title('Scatterplot of Ash & SuperPlastic')
sns.scatterplot(x='ash', y='coarseagg', data=dataset).set_title('Scatterplot of Ash & Coarseagg')
sns.scatterplot(x='ash', y='fineagg', data=dataset).set_title('Scatterplot of Ash & Fineagg')
sns.scatterplot(x='ash', y='age', data=dataset).set_title('Scatterplot of Ash & Age')
sns.scatterplot(x='ash', y='strength', data=dataset).set_title('Scatterplot of Ash & Strength')

sns.scatterplot(x='water', y='superplastic', data=dataset).set_title('Scatterplot of Water & SuperPlastic')
sns.scatterplot(x='water', y='coarseagg', data=dataset).set_title('Scatterplot of Water & Coarseagg')
sns.scatterplot(x='water', y='fineagg', data=dataset).set_title('Scatterplot of Water & Fineagg')
sns.scatterplot(x='water', y='age', data=dataset).set_title('Scatterplot of Water & Age')
sns.scatterplot(x='water', y='strength', data=dataset).set_title('Scatterplot of Water & Strength')

sns.scatterplot(x='superplastic', y='coarseagg', data=dataset).set_title('Scatterplot of SuperPlastic & Coarseagg')
sns.scatterplot(x='superplastic', y='fineagg', data=dataset).set_title('Scatterplot of SuperPlastic & Fineagg')
sns.scatterplot(x='superplastic', y='age', data=dataset).set_title('Scatterplot of SuperPlastic & Age')
sns.scatterplot(x='superplastic', y='strength', data=dataset).set_title('Scatterplot of SuperPlastic & Strength')

sns.scatterplot(x='coarseagg', y='fineagg', data=dataset).set_title('Scatterplot of Coarseagg & Fineagg')
sns.scatterplot(x='coarseagg', y='age', data=dataset).set_title('Scatterplot of Coarseagg & Age')
sns.scatterplot(x='coarseagg', y='strength', data=dataset).set_title('Scatterplot of Coarseagg & Strength')

sns.scatterplot(x='fineagg', y='age', data=dataset).set_title('Scatterplot of Fineagg & Age')
sns.scatterplot(x='fineagg', y='strength', data=dataset).set_title('Scatterplot of Fineagg & Strength')

sns.scatterplot(x='age', y='strength', data=dataset).set_title('Scatterplot of Age & Strength')

sns.pairplot(dataset)
sns.pairplot(dataset, diag_kind = 'kde')

# Heatmap
corr = dataset.corr()
sns.heatmap(corr, annot=True)

# Outliers
from scipy.stats import zscore
Z = np.abs(zscore(dataset))
print(np.where(Z>3))
print(Z[232][7])
df_out = dataset[(Z<3).all(axis=1)] # 49 outliers are removed

# Splitting the data into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df_out.drop('strength', axis=1), df_out['strength'], test_size=0.25, random_state=0)

# Normalization
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# Building the ANN model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
# Adding input layer and hidden layer
model.add(Dense(60, input_dim=8, kernel_initializer='normal', activation='relu'))
# Adding 2nd hidden layer
model.add(Dense(40, activation='relu'))
# Adding 3rd hidden layer
model.add(Dense(20, activation='relu'))
# Adding 4th hidden layer
model.add(Dense(10, activation='relu'))
# Adding output layer
model.add(Dense(1, activation='relu'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Fitting the model on data
model.fit((X_train), (Y_train), epochs=100, batch_size=100)

# Predicting the results on train set
pred_train = model.predict(X_train)
pred_train = pd.Series([i[0] for i in pred_train])
train_error = Y_train - pred_train
train_rmse = np.sqrt(np.mean(train_error**2))
print(train_rmse) # 24.10

plt.plot(pred_train,Y_train,"ro")
np.corrcoef(pred_train,Y_train) # we got high correlation 

# Predicting the results on test set
pred_test = model.predict(X_test)
pred_test = pd.Series([i[0] for i in pred_test])
test_error = Y_test - pred_test
test_rmse = np.sqrt(np.mean(test_error**2))
print(test_rmse) # 21.21

plt.plot(pred_test,Y_test,"bo")
np.corrcoef(pred_test,Y_test) # we got high correlation 

# Visualize the model
import keras, pydotplus, pydot
from keras.utils.vis_utils import model_to_dot, plot_model
keras.utils.vis_utils.pydot = pydot

plot_model(model, to_file='Neural Network Concrete.png')

# Another model from sklearn
from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(hidden_layer_sizes=(8,64)) # 56
regressor.fit(X_train, Y_train)

# Predicting the results on train set
pred_train = regressor.predict(X_train)
train_error = Y_train - pred_train
train_rmse = np.sqrt(np.mean(train_error**2))
print(train_rmse) # 9.91

# Predicting the results on test set
pred_test = regressor.predict(X_test)
test_error = Y_test - pred_test
test_rmse = np.sqrt(np.mean(test_error**2))
print(test_rmse) # 10.43
