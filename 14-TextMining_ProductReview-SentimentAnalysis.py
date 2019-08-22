# Reset the console
%reset -f

# Import libraries
# Importing requests to extract content from a url
import requests
from bs4 import BeautifulSoup as bs # for web scraping

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Creating empty review list
poco_reviews = []

for i in range(1,1000):
    phone = []
    url = "https://www.flipkart.com/poco-f1-xiaomi-steel-blue-128-gb/product-reviews/itmf8hu3fb9bjqz4?pid=MOBF85V7KKANFFZX&page="+str(i)
    response = requests.get(url)
    # creating soup object to iterate over the extracted content
    soup = bs(response.content, "html.parser")
    # Extract the content under the specific tag
    reviews = soup.find_all("div", attrs={'class':''})
    for i in range(len(reviews)):
        phone.append(reviews[i].text)
    # Adding the reviews of one page to empty list which in future contains all the reviews
    poco_reviews = poco_reviews + phone
    
# Writing reviews in a text file
with open('poco.txt','w', encoding = 'utf8') as output:
    output.write(str(poco_reviews))

# Joining all the reviews into single paragraph
poco_rev_string = " ".join(poco_reviews)

# Remove unwanted symbols from text 
import re
poco_rev_string = re.sub('[^a-zA-Z]', ' ', poco_rev_string)
# Remove numbers
poco_rev_string = re.sub('[0-9]', ' ', poco_rev_string)
# Convert to lower case
poco_rev_string = poco_rev_string.lower()

# Download the Stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
# split review into words
rev_words = poco_rev_string.split()

rev_words = [word for word in rev_words if not word in set(stopwords.words('english'))]

# Steaming words
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
rev_words = [ps.stem(word) for word in rev_words]

# Joining the words to string
rev_words = ' '.join(rev_words)

# Word Cloud can be performed on the string inputs. So combine all the words
wordcloud_poco = WordCloud(background_color='black',
                           width = 1800, height = 1400).generate(rev_words)
plt.imshow(wordcloud_poco)

# Creating Positive words and Negative Words
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
pos_word_list=[]
neu_word_list=[]
neg_word_list=[]

for word in rev_words:
    if (sid.polarity_scores(word)['compound']) >= 0.5:
        pos_word_list.append(word)
    elif (sid.polarity_scores(word)['compound']) <= -0.5:
        neg_word_list.append(word)
    else:
        neu_word_list.append(word)                

print('Positive :',pos_word_list)
print('Neutral :',neu_word_list)
print('Negative :',neg_word_list)

# negative word cloud
# Choosing the only words which are present in negwords
poco_neg_rev = " ".join ([word for word in rev_words if word in neg_word_list])

wordcloud_neg_rev = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(poco_neg_rev)

plt.imshow(wordcloud_neg_rev)

# Positive word cloud
# Choosing the only words which are present in positive words
poco_pos_rev = " ".join ([word for word in rev_words if word in pos_word_list])

wordcloud_pos_rev = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(poco_pos_rev)

plt.imshow(wordcloud_pos_rev)

# Unique words 
poco_unique_words = list(set(" ".join(poco_reviews).split(" ")))

#from textblob import TextBlob
#import textblob
#from textblob.sentiments import NaiveBayesAnalyzer
#import nltk
#nltk.download('movie_reviews')
#nltk.download('punkt')
#
#sent = TextBlob(rev_words)
#polarity = sent.sentiment.polarity
#subjectivity = sent.sentiment.subjectivity
#
#sent = TextBlob(rev_words, analyzer=NaiveBayesAnalyzer())
#classification = sent.sentiment.classification
#positive = sent.sentiment.p_pos
#negative = sent.sentiment.p_neg
#
#print(polarity, subjectivity, classification, positive, negative)

# Apply sentiment Analysis on each review
for review in poco_reviews:
    print(review)
    analysis = TextBlob(review)
    print(analysis.sentiment,"\n")


