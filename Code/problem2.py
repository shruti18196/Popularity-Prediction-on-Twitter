"""
 EE 219 Project 5 Problem 2
 Name: Weikun Han
 Date: 3/16/2017
 Reference:
  - https://google.github.io/styleguide/pyguide.html
  - https://arxiv.org/abs/1401.2018
  - https://ucla.box.com/s/nv9td9kvvfvg3tya0dlvbs1kn5o87gmv
  - https://dev.twitter.com/docs
  - http://statsmodels.sourceforge.net/
 Description:
  - Linear Regression model Using 5 Features
  - Ordinary Least Squares (OLS) Method
"""

from __future__ import print_function
from sets import Set
import os
import json
import datetime
import statsmodels.api as sm

# Use Twitter Developer Documentation python API to get tweet_traindata
def get_tweet_traindata(file_id):
    
    # This taindata is many tweets with 5 features
    tweet_traindata = {}
    
    # Initial information
    hourly_userid_list = {}
    filename = os.path.join(folder, dataset[file_id])
    
    # Open one file in the dataset under the folder
    with open(filename) as fp: 
        
        # Check each tweet in selected file in the dataset
        for line in fp:
            tweet = json.loads(line) 
            
            # Get the userid for who posted this tweet          
            userid = tweet["tweet"]["user"]["id"] 
            
            # Get the number of retweets for this tweet         
            retweets = tweet["metrics"]["citations"]["total"]
            
            # Get the number of followers for this tweet           
            followers = tweet["author"]["followers"]
            
            # Store the time when the tweet was posted hourly
            post_time = tweet["firstpost_date"] 
            post_time = datetime.datetime.fromtimestamp(post_time) 
            post_time_modified = datetime.datetime(post_time.year, 
                                                   post_time.month, 
                                                   post_time.day, 
                                                   post_time.hour, 
                                                   0, 
                                                   0)
            
            # Convert post_time into string in order use the dictionary nesting
            post_time_modified = unicode(post_time_modified)
            
            # Check if a new hour, and initial 5 features with zeros            
            if post_time_modified not in tweet_traindata:
                tweet_traindata[post_time_modified] = {'tweets_count' : 0, 
                                                       'retweets_count' : 0, 
                                                       'followers_count' : 0, 
                                                       'max_followers' : 0, 
                                                       'time' : 0}
                hourly_userid_list[post_time_modified] = Set([])
            
            # Keep count 3 features for the same time (hourly)
            tweet_traindata[post_time_modified]['tweets_count'] += 1
            tweet_traindata[post_time_modified]['retweets_count'] += retweets
            tweet_traindata[post_time_modified]['time'] = post_time.hour
            
            # Check if a userid is in hourly_userid_list
            if userid not in hourly_userid_list[post_time_modified]:
                hourly_userid_list[post_time_modified].add(userid)
                
                # Keep count 1 features for the same time (hourly)
                tweet_traindata[post_time_modified]['followers_count'] += followers
                
                # Update maximum number of followers of the users posting the hashtag
                if followers > tweet_traindata[post_time_modified]['max_followers']:
                
                    # Keep count 1 features for the same time (hourly)
                    tweet_traindata[post_time_modified]['max_followers'] = followers
    #print(tweet_traindata)
    return tweet_traindata

# Get tweet_testdata and tweet_predictdata (need work on)
def traindata_processing(hour):
    start_time = min(hour.keys()) 
    end_time = max(hour.keys())    
    predictors = []
    labels = []
    cur_hour = start_time
    while cur_hour <= end_time: 
        next_hour_tweet_count = 
        next_hour = cur_hour+datetime.timedelta(hours=1) 
        if next_hour in hour:
            next_hour_tweet_count = hour[next_hour]['tweets_count'] 
        if cur_hour in hour:
            predictors.append(hour[cur_hour].values()) 
            labels.append([next_hour_tweet_count])
        else: 
            temp = {'tweets_count':0, 'retweets_count':0, 'followers':0, 'max_followers':0, 'time':cur_hour.hour}    
            predictors.append(temp.values())
            labels.append([next_hour_tweet_count])
        cur_hour = next_hour
    return predictors, labels

# Print information
print(__doc__)
print()
    
# Input hashtags we want put in to the dataset under the folder
folder = "tweet_data"
dataset = ["tweets_#gohawks.txt", 
           "tweets_#gopatriots.txt", 
           "tweets_#nfl.txt", 
           "tweets_#patriots.txt", 
           "tweets_#sb49.txt", 
           "tweets_#superbowl.txt"]
hashtag_list = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]

# Extrect each dataset to do the data processing
for i in range(len(dataset)):
    tweet_traindata = get_tweet_traindata(i)
    
    # Modifie the dictionary nesting with 5 features
    tweet_traindata_modified = {}
    
    # Convert string index in tweet_traindata to time index in order to add a hour get the next hour
    for value in tweet_traindata:  
        current_time = datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        features = tweet_traindata[value]
        tweet_traindata_modified[current_time] = features

    # Get tweet_testdata and tweet_predictdata
    tweet_predictdata, tweet_testdata = traindata_processing(tweet_traindata_modified)
    
    # Use Statsmodels python API to do linear regression models analysis
    tweet_predictdata = sm.add_constant(tweet_predictdata)
    model = sm.OLS(tweet_testdata, tweet_predictdata)
    results = model.fit()
    
    # Print information
    print("-------------------------Processing Finshed %d---------------------------" % (i + 1))
    print(results.summary())
    print()
