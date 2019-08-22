# Reset the console
%reset -f

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
forfire = pd.read_csv('forestfires.csv')
forfire.head()
# Droping unnecessary columns
dataset = forfire.drop(['month','day'], axis=1)
dataset.head()
dataset.columns
dataset.dtypes
dataset.info()
dataset.shape

dataset.isnull().sum() # No missing values
dataset.drop_duplicates(keep='first', inplace=True) # 8 duplicates removed
dataset['size_category'].value_counts()

# Statistical Description
dataset.describe()
# Measures of Dispersion
np.var(dataset)
np.std(dataset)
# Skewness and Kurtosis
from scipy.stats import skew, kurtosis
skew(dataset.drop('size_category', axis=1))
kurtosis(dataset.drop('size_category', axis=1))

#Histogram
plt.hist(dataset['FFMC']);plt.title('Histogram of FFMC');plt.xlabel('FFMC');plt.ylabel('Frequency')
plt.hist(dataset['DMC'], color='coral');plt.title('Histogram of DMC');plt.xlabel('DMC');plt.ylabel('Frequency')
plt.hist(dataset['DC'], color='skyblue');plt.title('Histogram of DC');plt.xlabel('DC');plt.ylabel('Frequency')
plt.hist(dataset['ISI'], color='orange');plt.title('Histogram of ISI');plt.xlabel('ISI');plt.ylabel('Frequency')
plt.hist(dataset['temp'], color='brown');plt.title('Histogram of temp');plt.xlabel('temp');plt.ylabel('Frequency')
plt.hist(dataset['RH'], color='violet');plt.title('Histogram of RH');plt.xlabel('RH');plt.ylabel('Frequency')
plt.hist(dataset['wind'], color='teal');plt.title('Histogram of wind');plt.xlabel('wind');plt.ylabel('Frequency')
plt.hist(dataset['rain'], color='purple');plt.title('Histogram of rain');plt.xlabel('rain');plt.ylabel('Frequency')
plt.hist(dataset['area'], color='lightgreen');plt.title('Histogram of area');plt.xlabel('area');plt.ylabel('Frequency')

# Barplot 
import seaborn as sns
sns.countplot(forfire['month']).set_title('Count of Month')
sns.countplot(forfire['day']).set_title('Count of Day')
sns.countplot(x='month', hue='size_category', data=forfire).set_title('Count of Area Size by Month')
sns.countplot(x='day', hue='size_category', data=forfire).set_title('Count of Area Size by Day')

# Normal Q-Q plot
plt.plot(dataset.drop('size_category', axis=1));plt.legend(list(dataset.columns))

ffmc = np.array(dataset['FFMC'])
dmc = np.array(dataset['DMC'])
dc = np.array(dataset['DC'])
isi = np.array(dataset['ISI'])
temp = np.array(dataset['temp'])
rh = np.array(dataset['RH'])
wind = np.array(dataset['wind'])
rain = np.array(dataset['rain'])
area = np.array(dataset['area'])

from scipy import stats
stats.probplot(ffmc, dist='norm', plot=plt);plt.title('Probability plot of FFMC')
stats.probplot(dmc, dist='norm', plot=plt);plt.title('Probability plot of DMC')
stats.probplot(dc, dist='norm', plot=plt);plt.title('Probability plot of DC')
stats.probplot(isi, dist='norm', plot=plt);plt.title('Probability plot of ISI')
stats.probplot(temp, dist='norm', plot=plt);plt.title('Probability plot of temp')
stats.probplot(rh, dist='norm', plot=plt);plt.title('Probability plot of RH')
stats.probplot(wind, dist='norm', plot=plt);plt.title('Probability plot of wind')
stats.probplot(rain, dist='norm', plot=plt);plt.title('Probability plot of rain')
stats.probplot(area, dist='norm', plot=plt);plt.title('Probability plot of area')

# Normal Probability Distribution
x_ffmc = np.linspace(np.min(ffmc), np.max(ffmc))
y_ffmc = stats.norm.pdf(x_ffmc, np.median(x_ffmc), np.std(x_ffmc))
plt.plot(x_ffmc, y_ffmc);plt.xlim(np.min(x_ffmc), np.max(x_ffmc));plt.title('Normal Probability Distribution of FFMC');plt.xlabel('FFMC');plt.ylabel('Probability')

x_dmc = np.linspace(np.min(dmc), np.max(dmc))
y_dmc = stats.norm.pdf(x_dmc, np.median(x_dmc), np.std(x_dmc))
plt.plot(x_dmc, y_dmc);plt.xlim(np.min(x_dmc), np.max(x_dmc));plt.title('Normal Probability Distribution of DMC');plt.xlabel('DMC');plt.ylabel('Probability')

x_dc = np.linspace(np.min(dc), np.max(dc))
y_dc = stats.norm.pdf(x_dc, np.median(x_dc), np.std(x_dc))
plt.plot(x_dc, y_dc);plt.xlim(np.min(x_dc), np.max(x_dc));plt.title('Normal Probability Distribution of DC');plt.xlabel('DC');plt.ylabel('Probability')

x_isi = np.linspace(np.min(isi), np.max(isi))
y_isi = stats.norm.pdf(x_isi, np.median(x_isi), np.std(x_isi))
plt.plot(x_isi, y_isi);plt.xlim(np.min(x_isi), np.max(x_isi));plt.title('Normal Probability Distribution of ISI');plt.xlabel('ISI');plt.ylabel('Probability')

x_temp = np.linspace(np.min(temp), np.max(temp))
y_temp = stats.norm.pdf(x_temp, np.median(x_temp), np.std(x_temp))
plt.plot(x_temp, y_temp);plt.xlim(np.min(x_temp), np.max(x_temp));plt.title('Normal Probability Distribution of Temperature');plt.xlabel('Temperature');plt.ylabel('Probability')

x_rh = np.linspace(np.min(rh), np.max(rh))
y_rh = stats.norm.pdf(x_rh, np.median(x_rh), np.std(x_rh))
plt.plot(x_rh, y_rh);plt.xlim(np.min(x_rh), np.max(x_rh));plt.title('Normal Probability Distribution of RH');plt.xlabel('RH');plt.ylabel('Probability')

x_wind = np.linspace(np.min(wind), np.max(wind))
y_wind = stats.norm.pdf(x_wind, np.median(x_wind), np.std(x_wind))
plt.plot(x_wind, y_wind);plt.xlim(np.min(x_wind), np.max(x_wind));plt.title('Normal Probability Distribution of Wind');plt.xlabel('Wind');plt.ylabel('Probability')

x_rain = np.linspace(np.min(rain), np.max(rain))
y_rain = stats.norm.pdf(x_rain, np.median(x_rain), np.std(x_rain))
plt.plot(x_rain, y_rain);plt.xlim(np.min(x_rain), np.max(x_rain));plt.title('Normal Probability Distribution of Rain');plt.xlabel('Rain');plt.ylabel('Probability')

x_area = np.linspace(np.min(area), np.max(area))
y_area = stats.norm.pdf(x_area, np.median(x_area), np.std(x_area))
plt.plot(x_area, y_area);plt.xlim(np.min(x_area), np.max(x_area));plt.title('Normal Probability Distribution of Area');plt.xlabel('Area');plt.ylabel('Probability')

# Boxplot 
sns.boxplot(dataset['FFMC'],orient='v').set_title('Boxplot of FFMC')
sns.boxplot(dataset['DMC'], orient='v', color='coral').set_title('Boxplot of DMC')
sns.boxplot(dataset['DC'], orient='v', color='skyblue').set_title('Boxplot of DC')
sns.boxplot(dataset['ISI'], orient='v', color='orange').set_title('Boxplot of ISI')
sns.boxplot(dataset['temp'], orient='v', color='teal').set_title('Boxplot of temp')
sns.boxplot(dataset['RH'], orient='v', color='brown').set_title('Boxplot of RH')
sns.boxplot(dataset['wind'], orient='v', color='violet').set_title('Boxplot of wind')
sns.boxplot(dataset['rain'], orient='v', color='purple').set_title('Boxplot of rain')
sns.boxplot(dataset['area'], orient='v', color='lightgreen').set_title('Boxplot of area')

# Boxplot wrt categorical data
sns.boxplot(x='month', y='FFMC', data=forfire).set_title('Boxplot of Month & FFMC')
sns.boxplot(x='month', y='DMC', data=forfire).set_title('Boxplot of Month & DMC')
sns.boxplot(x='month', y='DC', data=forfire).set_title('Boxplot of Month & DC')
sns.boxplot(x='month', y='ISI', data=forfire).set_title('Boxplot of Month & ISI')
sns.boxplot(x='month', y='temp', data=forfire).set_title('Boxplot of Month & Temperature')
sns.boxplot(x='month', y='RH', data=forfire).set_title('Boxplot of Month & RH')
sns.boxplot(x='month', y='wind', data=forfire).set_title('Boxplot of Month & Wind')
sns.boxplot(x='month', y='rain', data=forfire).set_title('Boxplot of Month & Rain')
sns.boxplot(x='month', y='area', data=forfire).set_title('Boxplot of Month & Area')

# Scatterplot
sns.scatterplot(x='FFMC', y='DMC', data=forfire).set_title('Scatterplot of FFMC & DMC')
sns.scatterplot(x='FFMC', y='DC', data=forfire).set_title('Scatterplot of FFMC & DC')
sns.scatterplot(x='FFMC', y='ISI', data=forfire).set_title('Scatterplot of FFMC & ISI')
sns.scatterplot(x='FFMC', y='temp', data=forfire).set_title('Scatterplot of FFMC & Temperature')
sns.scatterplot(x='FFMC', y='RH', data=forfire).set_title('Scatterplot of FFMC & RH')
sns.scatterplot(x='FFMC', y='wind', data=forfire).set_title('Scatterplot of FFMC & Wind')
sns.scatterplot(x='FFMC', y='rain', data=forfire).set_title('Scatterplot of FFMC & Rain')
sns.scatterplot(x='FFMC', y='area', data=forfire).set_title('Scatterplot of FFMC & Area')

sns.scatterplot(x='DMC', y='DC', data=forfire).set_title('Scatterplot of DMC & DC')
sns.scatterplot(x='DMC', y='ISI', data=forfire).set_title('Scatterplot of DMC & ISI')
sns.scatterplot(x='DMC', y='temp', data=forfire).set_title('Scatterplot of DMC & Temperature')
sns.scatterplot(x='DMC', y='RH', data=forfire).set_title('Scatterplot of DMC & RH')
sns.scatterplot(x='DMC', y='wind', data=forfire).set_title('Scatterplot of DMC & Wind')
sns.scatterplot(x='DMC', y='rain', data=forfire).set_title('Scatterplot of DMC & Rain')
sns.scatterplot(x='DMC', y='area', data=forfire).set_title('Scatterplot of DMC & Area')

sns.scatterplot(x='DC', y='ISI', data=forfire).set_title('Scatterplot of DC & ISI')
sns.scatterplot(x='DC', y='temp', data=forfire).set_title('Scatterplot of DC & Temperature')
sns.scatterplot(x='DC', y='RH', data=forfire).set_title('Scatterplot of DC & RH')
sns.scatterplot(x='DC', y='wind', data=forfire).set_title('Scatterplot of DC & Wind')
sns.scatterplot(x='DC', y='rain', data=forfire).set_title('Scatterplot of DC & Rain')
sns.scatterplot(x='DC', y='area', data=forfire).set_title('Scatterplot of DC & Area')

sns.scatterplot(x='ISI', y='temp', data=forfire).set_title('Scatterplot of ISI & Temperature')
sns.scatterplot(x='ISI', y='RH', data=forfire).set_title('Scatterplot of ISI & RH')
sns.scatterplot(x='ISI', y='wind', data=forfire).set_title('Scatterplot of ISI & Wind')
sns.scatterplot(x='ISI', y='rain', data=forfire).set_title('Scatterplot of ISI & Rain')
sns.scatterplot(x='ISI', y='area', data=forfire).set_title('Scatterplot of ISI & Area')

sns.scatterplot(x='temp', y='RH', data=forfire).set_title('Scatterplot of temp & RH')
sns.scatterplot(x='temp', y='wind', data=forfire).set_title('Scatterplot of temp & Wind')
sns.scatterplot(x='temp', y='rain', data=forfire).set_title('Scatterplot of temp & Rain')
sns.scatterplot(x='temp', y='area', data=forfire).set_title('Scatterplot of temp & Area')

sns.scatterplot(x='RH', y='wind', data=forfire).set_title('Scatterplot of RH & Wind')
sns.scatterplot(x='RH', y='rain', data=forfire).set_title('Scatterplot of RH & Rain')
sns.scatterplot(x='RH', y='area', data=forfire).set_title('Scatterplot of RH & Area')

sns.scatterplot(x='wind', y='rain', data=forfire).set_title('Scatterplot of Wind & Rain')
sns.scatterplot(x='wind', y='area', data=forfire).set_title('Scatterplot of Wind & Area')

sns.scatterplot(x='rain', y='area', data=forfire).set_title('Scatterplot of Rain & Area')

#sns.pairplot(dataset)
#sns.pairplot(dataset, diag_kind='kde')
#sns.pairplot(dataset, hue='size_category')

# Heatmap
corr = dataset.corr()
sns.heatmap(corr, annot=True)

# Outliers
from scipy import stats
Z = np.abs(stats.zscore(dataset.drop('size_category', axis=1)))
print(np.where(Z>3))
print(Z[40][19])
df_out = dataset[(Z<3).all(axis=1)] # 119 outliers are removed

# Convert the categorical to binary class 
# small=0 and large = 1
#dataset.loc[dataset['size_category'] == 'small', 'size_category'] = 0
#dataset.loc[dataset['size_category'] == 'large', 'size_category'] = 1

# Creating dummy variables for size_category
df_out = pd.get_dummies(df_out)

#df_out.loc[dataset['size_category'] == 'small', 'size_category'] = 0
#df_out.loc[dataset['size_category'] == 'large', 'size_category'] = 1
#df_out['size_category'].value_counts() # 75, 25

# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df_out.drop('area',axis=1),df_out['area'], test_size=0.30, random_state=0)

# Normalization
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# Creating the ANN model
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
# Adding the input layer and hidden layer
model.add(Dense(50, input_dim=29, kernel_initializer='normal', activation='relu'))
# Adding 2nd hidden layer
model.add(Dense(40, activation='relu'))
# Adding 3rd hidden layer
model.add(Dense(20, activation='relu'))
# Adding 4th hidden layer
model.add(Dense(10, activation='relu'))
# Adding 5th hidden layer
model.add(Dense(3, activation='relu'))
# Adding output layer
model.add(Dense(1, activation='relu'))
# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Fiiting the ANN model on trainset data
model.fit((X_train), (Y_train), epochs=100, batch_size=100)

# Scores
score = model.evaluate(X_train, Y_train)
print(score)

# Predicting the results on train set
pred_train = model.predict(X_train)
pred_train = [i[0] for i in pred_train]
train_error = Y_train - pred_train
train_rmse = np.sqrt(np.mean(train_error**2))
print(train_rmse) # 11.28

# Predicting the results on testset 
pred_test = model.predict(X_test)
pred_test = [i[0] for i in pred_test]
test_error = Y_test - pred_test
test_rmse = np.sqrt(np.mean(test_error**2))
print(test_rmse) # 21.04

# Plot the ANN model
from ann_visualizer.visualize import ann_viz
ann_viz(model, title='NeuralNetwork for ForestFires')

# Another model from sklearn
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(28,67))
mlp.fit(X_train, Y_train)

# Predicting the results on train set
pred_train = mlp.predict(X_train)
train_error = Y_train - pred_train
train_rmse = np.sqrt(np.mean(train_error**2))
print(train_rmse) # 30.76

# Predicting the results on test set
pred_test = mlp.predict(X_test)
test_error = Y_test - pred_test
test_rmse = np.sqrt(np.mean(test_error**2))
print(test_rmse) # 12.32
