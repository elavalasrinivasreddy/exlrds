# Reset the console
%reset -f

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
dataset = pd.read_csv('iris.csv')
dataset.head()
dataset.info()

dataset.columns
dataset.dtypes
dataset['Species'].value_counts()

dataset.isnull().sum() # No missing values
dataset.shape
dataset.drop_duplicates(keep='first', inplace=True) # One Duplicate removed

# Statistical Description
dataset.describe()
# Measures of Dispersion
np.var(dataset)
np.std(dataset)
# Skewness and Kurtosis
from scipy.stats import skew, kurtosis
skew(dataset.drop('Species', axis=1))
kurtosis(dataset.drop('Species', axis=1))

# Histogram
plt.hist(dataset['Sepal.Length']);plt.title('Histogram of Sepal Length');plt.xlabel('Sepal Length');plt.ylabel('Frequency')
plt.hist(dataset['Sepal.Width'], color='coral');plt.title('Histogram of Sepal Width');plt.xlabel('Sepal Width');plt.ylabel('Frequency')
plt.hist(dataset['Petal.Length'], color='teal');plt.title('Histogram of Petal Length');plt.xlabel('Petal Length');plt.ylabel('Frequency')
plt.hist(dataset['Petal.Width'], color='purple');plt.title('Histogram of Petal Width');plt.xlabel('Petal Width');plt.ylabel('Frequency')

# Barplot 
import seaborn as sns
sns.countplot(dataset['Species']).set_title('Count of Species')

# Normal Q-Q plot
plt.plot(dataset.drop('Species', axis=1));plt.legend(list(dataset.columns))

sl = np.array(dataset['Sepal.Length'])
sw = np.array(dataset['Sepal.Width'])
pl = np.array(dataset['Petal.Length'])
pw = np.array(dataset['Petal.Width'])

from scipy import stats
stats.probplot(sl, dist='norm', plot=plt);plt.title('Probability plot of Sepal Length')
stats.probplot(sw, dist='norm', plot=plt);plt.title('Probability plot of Sepal Width')
stats.probplot(pl, dist='norm', plot=plt);plt.title('Probability plot of Petal Length')
stats.probplot(pw, dist='norm', plot=plt);plt.title('Probability plot of Petal Width')

# Normal Probability Distribution
x_sl = np.linspace(np.min(sl), np.max(sl))
y_sl = stats.norm.pdf(x_sl, np.median(x_sl), np.std(x_sl))
plt.plot(x_sl, y_sl);plt.xlim(np.min(sl), np.max(sl));plt.title('Normal Probability Distribution of Sepal Length');plt.xlabel('Sepal Length');plt.ylabel('Probability')

x_sw = np.linspace(np.min(sw), np.max(sw))
y_sw = stats.norm.pdf(x_sw, np.median(x_sw), np.std(x_sw))
plt.plot(x_sw, y_sw);plt.xlim(np.min(sw), np.max(sw));plt.title('Normal Probability Distribution of Sepal Width');plt.xlabel('Sepal Width');plt.ylabel('Probability')

x_pl = np.linspace(np.min(pl), np.max(pl))
y_pl = stats.norm.pdf(x_pl, np.median(x_pl), np.std(x_pl))
plt.plot(x_pl, y_pl);plt.xlim(np.min(pl), np.max(pl));plt.title('Normal Probability Distribution of Petal Length');plt.xlabel('Petal Length');plt.ylabel('Probability')

x_pw = np.linspace(np.min(pw), np.max(pw))
y_pw = stats.norm.pdf(x_pw, np.median(x_pw), np.std(x_pw))
plt.plot(x_pw, y_pw);plt.xlim(np.min(pw), np.max(pw));plt.title('Normal Probability Distribution of Sepal Length');plt.xlabel('Petal Width');plt.ylabel('Probability')

# Boxplot 
sns.boxplot(dataset['Sepal.Length'], orient='v').set_title('Boxplot of Sepal Length')
sns.boxplot(dataset['Sepal.Width'], orient='v', color='coral').set_title('Boxplot of Sepal Width')
sns.boxplot(dataset['Petal.Length'], orient='v', color='teal').set_title('Boxplot of Petal Length')
sns.boxplot(dataset['Petal.Width'], orient='v', color='purple').set_title('Boxplot of Petal Width')

# Boxplot wrt Species
sns.boxplot(x='Species', y='Sepal.Length', data = dataset).set_title('Boxplot of Species & Sepal Length')
sns.boxplot(x='Species', y='Sepal.Width', data = dataset).set_title('Boxplot of Species & Sepal Width')
sns.boxplot(x='Species', y='Petal.Length', data = dataset).set_title('Boxplot of Species & Petal Length')
sns.boxplot(x='Species', y='Petal.Width', data = dataset).set_title('Boxplot of Species & Petal Width')

# Scatterplot
sns.scatterplot(x='Sepal.Length', y='Sepal.Width', data=dataset).set_title('Scatterplot of Sepal Length & Sepal Width')
sns.scatterplot(x='Sepal.Length', y='Petal.Length', data=dataset).set_title('Scatterplot of Sepal Length & Petal Length')
sns.scatterplot(x='Sepal.Length', y='Petal.Width', data=dataset).set_title('Scatterplot of Sepal Length & Petal Width')

sns.scatterplot(x='Sepal.Width', y='Petal.Length', data=dataset).set_title('Scatterplot of Sepal Width & Petal Length')
sns.scatterplot(x='Sepal.Width', y='Petal.Width', data=dataset).set_title('Scatterplot of Sepal Width & Petal Width')

sns.scatterplot(x='Petal.Length', y='Petal.Width', data=dataset).set_title('Scatterplot of Petal Length & Petal Width')

sns.pairplot(dataset)
sns.pairplot(dataset, diag_kind='kde', hue='Species')

# Heatmap
corr = dataset.corr()
sns.heatmap(corr, annot=True)

# Outliers
from scipy.stats import zscore
Z = np.abs(zscore(dataset.drop('Species',axis=1)))
print(np.where(Z>3))
print(Z[15][1])
df_out = dataset[(Z<3).all(axis=1)] # 1 outlier removed

# Splitting the data into train-set and test-set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df_out.drop('Species', axis=1), df_out['Species'], test_size=0.3, random_state=0)

# Fitting the Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
en_model = DecisionTreeClassifier(criterion='entropy')
en_model.fit(X_train, Y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
# Predicting results on train-set data
pred_train = en_model.predict(X_train)
train_cm = confusion_matrix(Y_train, pred_train)
print(train_cm)
train_acc = accuracy_score(Y_train, pred_train)
print(train_acc) # 100%

# Predicting results on test-set data
pred_test = en_model.predict(X_test)
test_cm = confusion_matrix(Y_test, pred_test)
print(test_cm)
test_acc = accuracy_score(Y_test, pred_test)
print(test_acc) # 97%

# Fitting Decision Tree Classifier using 'gini'
gi_model = DecisionTreeClassifier(criterion='gini')
gi_model.fit(X_train, Y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
# Predicting results on train-set data
pred_train = gi_model.predict(X_train)
train_cm = confusion_matrix(Y_train, pred_train)
print(train_cm)
train_acc = accuracy_score(Y_train, pred_train)
print(train_acc) # 100%

# Predicting results on test-set data
pred_test = gi_model.predict(X_test)
test_cm = confusion_matrix(Y_test, pred_test)
print(test_cm)
test_acc = accuracy_score(Y_test, pred_test)
print(test_acc) # 95%

# Fitting Random Forest Classifier using 'entropy'
from sklearn.ensemble import RandomForestClassifier
iris_en_model = RandomForestClassifier(n_jobs=4, n_estimators=10, criterion='entropy')
iris_en_model.fit(X_train, Y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
# Predicting results on train-set data
train_pred = iris_en_model.predict(X_train)
train_cm = confusion_matrix(Y_train, train_pred)
print(train_cm)
train_acc = accuracy_score(Y_train, train_pred)
print(train_acc) # 100%

# Predicting results on test-set data
test_pred = iris_en_model.predict(X_test)
test_cm = confusion_matrix(Y_test, test_pred)
print(test_cm)
test_acc = accuracy_score(Y_test, test_pred)
print(test_acc) # 95%

# Fitting Random Forest Classifier using 'gini'
iris_gi_model = RandomForestClassifier(n_jobs=4, n_estimators=100, criterion='gini')
iris_gi_model.fit(X_train, Y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
# Predicting results on train-set data
train_pred = iris_gi_model.predict(X_train)
train_cm = confusion_matrix(Y_train, train_pred)
print(train_cm)
train_acc = accuracy_score(Y_train, train_pred)
print(train_acc) # 100%

# Predicting results on test-set data
test_pred = iris_gi_model.predict(X_test)
test_cm = confusion_matrix(Y_test, test_pred)
print(test_cm)
test_acc = accuracy_score(Y_test, test_pred)
print(test_acc) # 95%
