# Reset the console
%reset -f

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load dataset
Zoo = pd.read_csv('Zoo.csv')
Zoo.head() # All variables are binary
Zoo.shape
dataset = Zoo.iloc[:,1:] # Remove animal name
dataset.shape
dataset.head()
dataset.type.value_counts()

dataset.isnull().sum() # no missing values
#dataset.drop_duplicates(keep='first', inplace=True) # 42 rows are duplicate

# Piechart for binary data
import seaborn as sns
sns.countplot(dataset['hair']).set_title('Countplot of Hair')
sns.countplot(dataset['feathers']).set_title('Countplot of Feathers')
sns.countplot(dataset['eggs']).set_title('Countplot of Eggs')
sns.countplot(dataset['milk']).set_title('Countplot of Milk')
sns.countplot(dataset['airborne']).set_title('Countplot of Airborne')
sns.countplot(dataset['aquatic']).set_title('Countplot of Aquatic')
sns.countplot(dataset['predator']).set_title('Countplot of Predator')
sns.countplot(dataset['toothed']).set_title('Countplot of Toothed')
sns.countplot(dataset['backbone']).set_title('Countplot of Backbone')
sns.countplot(dataset['breathes']).set_title('Countplot of Breathes')
sns.countplot(dataset['venomous']).set_title('Countplot of Venomous')
sns.countplot(dataset['fins']).set_title('Countplot of Fins')
sns.countplot(dataset['legs']).set_title('Countplot of Legs')
sns.countplot(dataset['tail']).set_title('Countplot of Tail')
sns.countplot(dataset['domestic']).set_title('Countplot of Domestic')
sns.countplot(dataset['catsize']).set_title('Countplot of Catsize')
sns.countplot(dataset['type']).set_title('Countplot of Type of Animals')

# Boxplot
sns.boxplot(dataset['feathers'], orient='v').set_title('Boxplot of Feathers')
sns.boxplot(dataset['airborne'], orient='v').set_title('Boxplot of Airborne')
sns.boxplot(dataset['backbone'], orient='v').set_title('Boxplot of Backbone')
sns.boxplot(dataset['breathes'], orient='v').set_title('Boxplot of Breathes')
sns.boxplot(dataset['venomous'], orient='v').set_title('Boxplot of Venomous')
sns.boxplot(dataset['fins'], orient='v').set_title('Boxplot of Fins')
sns.boxplot(dataset['legs'], orient='v').set_title('Boxplot of Legs')
sns.boxplot(dataset['domestic'], orient='v').set_title('Boxplot of Domestic')
sns.boxplot(dataset['type'], orient='v').set_title('Boxplot of Type of Animals')

# Heatmap
sns.heatmap(dataset.corr(), annot=True)

# Metrics of features
X = dataset.iloc[:,0:-1].values
Y = dataset.iloc[:,-1].values


# Categorical data and dummy variables

# Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fiiting classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric ='minkowski',p=2)
classifier.fit(X_train, Y_train)

# Predicting train set results
train_acc = np.mean(classifier.predict(X_train)==Y_train)
train_acc  # 95%

# Predicting the Test set results
test_acc = np.mean(classifier.predict(X_test)==Y_test)
test_acc # 100%

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
accuracy = []
from sklearn.neighbors import KNeighborsClassifier as KNC
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(X_train,Y_train)
    train_acc = np.mean(neigh.predict(X_train)==Y_train)
    test_acc = np.mean(neigh.predict(X_test)==Y_test)
    accuracy.append([train_acc,test_acc])

import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in accuracy],"bo-")
# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in accuracy],"ro-")
plt.legend(["train","test"])

# n_neighbors =3 is the best for KNN
classifier = KNeighborsClassifier(n_neighbors=3,metric ='minkowski',p=2)
classifier.fit(X_train, Y_train)

# Predicting train set results
train_acc = np.mean(classifier.predict(X_train)==Y_train)
train_acc  # 96%

# Predicting the Test set results
test_acc = np.mean(classifier.predict(X_test)==Y_test)
test_acc # 95%
