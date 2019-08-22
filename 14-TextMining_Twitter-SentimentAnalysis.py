#Reset the console
%reset -f

# Import libraries
import tweepy
from textblob import TextBlob

# Consumer API Keys
consumer_key = 'uX2fjWa0mSdtwnH9iHfRhyB6e'
consumer_secret = '626PrESQXWfMskOJeVVW0J4fkvVDrcOOxvyaWpWvLC0eHl7z2l'

# Access token & Access token Secret
access_token = '2923008498-S5X2EblUqwlPLaMRU9mwb9uuwJKWD7Cos4ISmja'
access_token_secret = 'jfADfRCPfAz5HA2LquqAcJeNBg1n5u32mhGSmeKXYUFMV'

# Authentication with Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Getting Tweets from twitter
public_tweets = api.search('nithiin') # searching word 
# Apply sentiment Analysis on each tweet
for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)

#####################################################################################################################

# Another Method
# Import libraries
import pandas as pd
import  tweepy
import vaderSentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Twitter API authentication variables
consumer_key = 'uX2fjWa0mSdtwnH9iHfRhyB6a'
consumer_secret = '626PrESQXWfMskOJeVVW0J4fkvVDrcOOxvyaWpWvLC0eHl7z2n'
access_token = '2923008498-S5X2EblUqwlPLaMRU9mwb9uuwJKWD7Cos4ISmjs'
access_token_secret = 'jfADfRCPfAz5HA2LquqAcJeNBg1n5u32mhGSmeKXYUFMQ'
# Authentication with Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
# Enter the text and how many tweets you want 
tweets = api.search('Dhoni', count =400)
# Store the tweets into dataframe
data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns =['Tweets'])
data.head()
print(tweets[0].created_at)

import nltk
nltk.download('vader_lexicon')
# Applying Sentiment Analysis
sid = SentimentIntensityAnalyzer()
lists = [] # Creating list of tweets 

for index, row in data.iterrows():
    ss = sid.polarity_scores(row['Tweets'])
    lists.append(ss)

se = pd.Series(lists)
data['polarity'] = se.values # Sentiment values [-1 to 1]
data.head(100)

############################################################################################################

def get_all_tweets(text):
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    all_tweets = []
    new_tweets = api.user_timeline(text,count=1000)
    all_tweets.extend(new_tweets)
    
    old_tweets = all_tweets[-1].id - 1
    while len(new_tweets)>0:
        new_tweets = api.user_timeline(text,count=1000,max_id=old_tweets)
        #save most recent tweets
        all_tweets.extend(new_tweets)
        #update the id of the oldest tweet less one
        old_tweets = all_tweets[-1].id - 1
        print ("...%s tweets downloaded so far" % (len(all_tweets))) # tweet.get('user', {}).get('location', {})
 
    out_tweets = [[tweet.created_at,tweet.entities["hashtags"],tweet.entities["user_mentions"],tweet.favorite_count,
                  tweet.geo,tweet.id_str,tweet.lang,tweet.place,tweet.retweet_count,tweet.retweeted,tweet.source,tweet.text,
                  tweet._json["user"]["location"],tweet._json["user"]["name"],tweet._json["user"]["time_zone"],
                  tweet._json["user"]["utc_offset"]] for tweet in all_tweets]
    
    import pandas as pd
    tweets_df = pd.DataFrame(columns = ["time","hashtags","user_mentions","favorite_count",
                                    "geo","id_str","lang","place","retweet_count","retweeted","source",
                                    "text","location","name","time_zone","utc_offset"])
    tweets_df["time"]  = pd.Series([str(i[0]) for i in out_tweets])
    tweets_df["hashtags"] = pd.Series([str(i[1]) for i in out_tweets])
    tweets_df["user_mentions"] = pd.Series([str(i[2]) for i in out_tweets])
    tweets_df["favorite_count"] = pd.Series([str(i[3]) for i in out_tweets])
    tweets_df["geo"] = pd.Series([str(i[4]) for i in out_tweets])
    tweets_df["id_str"] = pd.Series([str(i[5]) for i in out_tweets])
    tweets_df["lang"] = pd.Series([str(i[6]) for i in out_tweets])
    tweets_df["place"] = pd.Series([str(i[7]) for i in out_tweets])
    tweets_df["retweet_count"] = pd.Series([str(i[8]) for i in out_tweets])
    tweets_df["retweeted"] = pd.Series([str(i[9]) for i in out_tweets])
    tweets_df["source"] = pd.Series([str(i[10]) for i in out_tweets])
    tweets_df["text"] = pd.Series([str(i[11]) for i in out_tweets])
    tweets_df["location"] = pd.Series([str(i[12]) for i in out_tweets])
    tweets_df["name"] = pd.Series([str(i[13]) for i in out_tweets])
    tweets_df["time_zone"] = pd.Series([str(i[14]) for i in out_tweets])
    tweets_df["utc_offset"] = pd.Series([str(i[15]) for i in out_tweets])
    tweets_df.to_csv(text+"_tweets.csv")
    return tweets_df

SushmaSwaraj = get_all_tweets("shushma swaraj")
tweets = pd.read_csv('shushma swaraj_tweets.csv')
