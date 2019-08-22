# Reset the console
%reset -f

# Import the libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Load the text data
dataset = pd.read_csv('sms_raw_NB.csv', encoding='ISO-8859-1')
dataset.head()
dataset.shape
dataset.info()
dataset.columns
dataset.isnull().sum() # No missing values
dataset.drop_duplicates(keep='first', inplace=True) # 603 rows are duplicates
dataset.dtypes
dataset.type.value_counts()

# Countplot of Type column
import seaborn as sns
sns.countplot(dataset['type']).set_title('Count of ham & spam')

# Cleaning the text data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Remove all punctuations, numbers
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w=[]
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

dataset['text'] = dataset['text'].apply(cleaning_text)

# Removing empty rows
dataset.shape
dataset = dataset.loc[dataset['text'] != " ", :]

# CountVectorizer
# convert a collection of text documents to a matrix of token counts

# TfidfTransformer
# Transform a count matrix to a normalized tf or tf-idf representation

# Creating a matrix of token counts for the entire text document 

def split_into_words(i):
    return [word for word in i.split(" ")]

# Splitting the data into train and test data sets
from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(dataset,test_size=0.25, random_state=0)

# Preparing sms text into word count matrix format
dataset_bow = CountVectorizer(analyzer=split_into_words).fit(dataset.text)

# For all sms
sms_matrix = dataset_bow.transform(dataset.text)
sms_matrix.shape # (5559, 6661)

# For training sms
train_sms_matrix = dataset_bow.transform(data_train.text)
train_sms_matrix.shape # (4169, 6661)

# For testing sms
test_sms_matrix = dataset_bow.transform(data_test.text)
test_sms_matrix.shape # (1390, 6661)

# Learning Term weighting and normalizing on entire sms
tfidf_transformer = TfidfTransformer().fit(sms_matrix)

# Preparing TFIDF for train sms
train_tfidf = tfidf_transformer.transform(train_sms_matrix)
train_tfidf.shape # (4169, 6661)

# Preparing TFIDF for test sms
test_tfidf = tfidf_transformer.transform(test_sms_matrix)
test_tfidf.shape # (1390, 6661)

# Preparing a Naive Bayes model on training data set
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

# Multinomial Naive Bayes classifier
classifier_mb = MB()
classifier_mb.fit(train_tfidf, data_train.type)
train_pred_mb = classifier_mb.predict(train_tfidf)
acc_train_mb = np.mean(train_pred_mb == data_train.type)
print(acc_train_mb) # 96%

test_pred_mb = classifier_mb.predict(test_tfidf)
acc_test_mb = np.mean(test_pred_mb == data_test.type)
print(acc_test_mb) # 96%


# Gaussian Navie Bayes classifier
classifier_gb = GB()
# We need to convert tfidf to array format which is compatible for GB.
classifier_gb.fit(train_tfidf.toarray(), data_train.type.values)
train_pred_gb = classifier_gb.predict(train_tfidf.toarray())
acc_train_gb = np.mean(train_pred_gb == data_train.type)
print(acc_train_gb) # 90%

test_pred_gb = classifier_gb.predict(test_tfidf.toarray())
acc_test_gb = np.mean(test_pred_gb == data_test.type)
print(acc_test_gb) # 84%


# Without TFIDF matrix

# Multinomial Navie Bayes classifier
classifier_mb = MB()
classifier_mb.fit(train_sms_matrix, data_train.type)
train_pred_mb = classifier_mb.predict(train_sms_matrix)
acc_train_mb = np.mean(train_pred_mb == data_train.type)
print(acc_train_mb) # 98%

test_pred_mb = classifier_mb.predict(test_sms_matrix)
acc_test_mb = np.mean(test_pred_mb == data_test.type)
print(acc_test_mb) # 97%


# Gaussian Navie Bayes classifier
classifier_gb = GB()
# We need to convert tfidf to array format which is compatible for GB.
classifier_gb.fit(train_sms_matrix.toarray(), data_train.type.values)
train_pred_gb = classifier_gb.predict(train_sms_matrix.toarray())
acc_train_gb = np.mean(train_pred_gb == data_train.type)
print(acc_train_gb) # 90%

test_pred_gb = classifier_gb.predict(test_sms_matrix.toarray())
acc_test_gb = np.mean(test_pred_gb == data_test.type)
print(acc_test_gb) # 84%
