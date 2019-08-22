# Reset the console
%reset -f

# Import libraries
from selenium import webdriver
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup as bs
from selenium.webdriver import ActionChains

from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotVisibleException
import time
import re
import matplotlib.pyplot as plt


# Open the Browser
browser = webdriver.Chrome(executable_path= 'D:\Downloads\chromedriver_win32\chromedriver.exe')
#browser = webdriver.Firefox(executable_path="D:\Downloads\geckodriver.exe")

# Url of the movie user review page
page = "https://www.imdb.com/title/tt3521164/reviews?ref_=tt_ql_3"
# Get the page in browser
browser.get(page)
# Create the empty list to store the reviews
reviews = []
i=1
# Loop for executing the load more button 
while (i>0):
    browser.execute_script('window.scrollTo(0,document.body.scrollHeight);')
    try:
        button = browser.find_element_by_class_name("ipl-load-more__button")
        print('Load More Button', button, '\n')
        ActionChains(browser).move_to_element(button).click(button).perform()
        print('Button is Clicked')
        time.sleep(5)
    # Continuously load the page until no "load more button"
    # You want only few reviews to break the loop 
    # press "Ctrl + c"
    except NoSuchElementException:
        break
    except ElementNotVisibleException:
        break
    
    # Every time you click yes to load the "load more button"
    
#    except Exception as e:
#        print('Click Error:', e)
#    # You want click y for more reviews under load more button
#    click = input('You Want More Click..? (y/n)')
#    if click !='y':
#        break
    
browser.close() # Close the browser

# get the page source using BeautifulSoup
soup = bs(browser.page_source, 'html.parser')
# Find the review content tag for user review
rev = soup.find_all('div', attrs={'class', 'text'})
# Loop for the each user review in content tag store in reviews list format
for i in range(len(rev)):reviews.append(rev[i].text)

# length of the reviews
len(reviews)

# writng reviews in a text file 
with open("review.txt","w",encoding='utf8') as output:output.write(str(reviews))

# Joinining all the reviews into single paragraph 
review_string = " ".join(reviews)

# Cleaning the text string
# Removing the symbols and special characters
review_string = re.sub("[^A-Za-z]"," ",review_string)
# Removing the numbers
review_string = re.sub("[0-9]"," ",review_string)
# Convert to lower case
review_string = review_string.lower()
# Remove unwanted space between the words in string
review_string = " ".join(review_string.split())

# Stop words
import nltk
nltk.download('stopwords') # Download stop words
from nltk.corpus import stopwords
# Split the review_string into words
review_words = review_string.split() # 117959 words 

review_words = [word for word in review_words if not word in set(stopwords.words('english'))] # 59726

# Steaming Words
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review_words = [ps.stem(word) for word in review_words]

# Join the words into string
movie_review = ' '.join(review_words)

# Word Cloud can be performed on the string inputs. So combine all the words
from wordcloud import WordCloud
wordcloud = WordCloud(background_color='black',width = 1800, height = 1400).generate(movie_review)
plt.imshow(wordcloud)

# Creating Positive words and Negative Words
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
pos_word_list=[]
neu_word_list=[]
neg_word_list=[]

for word in review_words:
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
movie_neg_rev = " ".join ([word for word in review_words if word in neg_word_list])

wordcloud_neg_rev = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(movie_neg_rev)

plt.imshow(wordcloud_neg_rev)

# Positive word cloud
# Choosing the only words which are present in positive words
movie_pos_rev = " ".join ([word for word in review_words if word in pos_word_list])

wordcloud_pos_rev = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(movie_pos_rev)

plt.imshow(wordcloud_pos_rev)

# Apply sentiment Analysis on each review
for review in reviews:
    print(review)
    analysis = TextBlob(review)
    print(analysis.sentiment,"\n")
    

from textblob import TextBlob
import textblob
from textblob.sentiments import NaiveBayesAnalyzer
import nltk
nltk.download('movie_reviews')
nltk.download('punkt')

sent = TextBlob(review_string)
polarity = sent.sentiment.polarity
subjectivity = sent.sentiment.subjectivity

sent = TextBlob(review_string, analyzer =NaiveBayesAnalyzer())
classification = sent.sentiment.classification
positive = sent.sentiment.p_pos
negative = sent.sentiment.p_neg

print(polarity, subjectivity, classification, positive, negative)

# Creating a data frame 
import pandas as pd
movie_reviews = pd.DataFrame(columns = ["reviews"])
movie_reviews["reviews"] = reviews

movie_reviews.to_csv("movie_reviews.csv",encoding="utf-8")
