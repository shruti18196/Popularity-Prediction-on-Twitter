"""
 EE 219 Project 5 Problem 3
 Name: Weikun Han
 Date: 3/16/2017
 Reference:
  - https://google.github.io/styleguide/pyguide.html
  - https://arxiv.org/abs/1401.2018
  - https://ucla.box.com/s/nv9td9kvvfvg3tya0dlvbs1kn5o87gmv
  - https://dev.twitter.com/docs
  - http://statsmodels.sourceforge.net/
 Description:
  - Linear Regression model Using 10 Features
  - Ordinary Least Squares (OLS) Method
"""

from __future__ import print_function
from sets import Set
import os
import json
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.figure as fig

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
            
            # Get the number of favorites for this tweet           
            favorites = tweet["tweet"]["favorite_count"]
            
            # Get the rank of tweet            
            rankingscore = tweet["metrics"]["ranking_score"]
            
            # Get the number of user mention for this tweet     
            usermentions = len(tweet["tweet"]["entities"]["user_mentions"])
            
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
            
            # Check if a new hour, and initial 10 features with zeros            
            if post_time_modified not in tweet_traindata:
                tweet_traindata[post_time_modified] = {'tweets_count' : 0, 
                                                       'retweets_count' : 0, 
                                                       'followers_count' : 0, 
                                                       'max_followers' : 0, 
                                                       'time' : 0,
                                                       'usermentions_count' : 0,            
                                                       'favorites_count' : 0,
                                                       'max_favorites' : 0,
                                                       'rankingscore' : 0,
                                                       'userid_count' : 0}
                hourly_userid_list[post_time_modified] = Set([])
            
            # Keep count 5 features for the same time (hourly)
            tweet_traindata[post_time_modified]['tweets_count'] += 1
            tweet_traindata[post_time_modified]['retweets_count'] += retweets
            tweet_traindata[post_time_modified]['time'] = post_time.hour
            tweet_traindata[post_time_modified]['usermentions_count'] += usermentions
            tweet_traindata[post_time_modified]['favorites_count'] += favorites
            tweet_traindata[post_time_modified]['rankingscore'] += rankingscore
            
            # Check if a is max favorite and update it
            if favorites > tweet_traindata[post_time_modified]['max_favorites']:
                
                # Keep count 1 features for the same time (hourly)
                tweet_traindata[post_time_modified]['max_favorites'] = favorites
            
            # Check if a userid is in hourly_userid_list
            if userid not in hourly_userid_list[post_time_modified]:
                hourly_userid_list[post_time_modified].add(userid)
                
                # Keep count 1 features for the same time (hourly)
                tweet_traindata[post_time_modified]['followers_count'] += followers
                
                # Keep count 1 features for the same time (hourly)
                tweet_traindata[post_time_modified]['userid_count'] += 1
                
                # Update maximum number of followers of the users posting the hashtag
                if followers > tweet_traindata[post_time_modified]['max_followers']:
                    
                    # Keep count 1 features for the same time (hourly)
                    tweet_traindata[post_time_modified]['max_followers'] = followers
    #print(tweet_traindata)
    return tweet_traindata

# need work on
def variables_lables_matrix(tweet_traindata):
    start_time = min(tweet_traindata.keys()) #Find the start and end time from the data
    end_time = max(tweet_traindata.keys())    
    predictors = []
    labels = []
    cur_hour = start_time
    while cur_hour <= end_time: #Keep looping from the start time till the end time
        next_hour_tweet_count = 0 #Initialize the label to be zero
        next_hour = cur_hour+datetime.timedelta(hours=1) #Go to the next hour
        if next_hour in tweet_traindata:
            next_hour_tweet_count = tweet_traindata[next_hour]['tweets_count'] #Update the label
        if cur_hour in tweet_traindata:
            predictors.append(tweet_traindata[cur_hour].values()) #Obtain the predictors
            labels.append([next_hour_tweet_count])
        else:
            temp = {'tweets_count' : 0, 
                    'retweets_count' : 0, 
                    'followers_count' : 0, 
                    'max_followers' : 0, 
                    'time' : cur_hour.hour,
                    'usermentions_count' : 0,            
                    'favorites_count' : 0,
                    'max_favorites' : 0,
                    'rankingscore' : 0,
                    'userid_count' : 0}
                    
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

# need work on
for i in range(len(dataset)):
    tweet_traindata = get_tweet_traindata(i)
    modified_hourwise_features = {}
    for time_value in tweet_traindata:
        cur_hour = datetime.datetime.strptime(time_value, "%Y-%m-%d %H:%M:%S")
        features = tweet_traindata[time_value]
        modified_hourwise_features[cur_hour] = features
    predictors, labels = variables_lables_matrix(modified_hourwise_features)
    predictors = sm.add_constant(predictors)    
    model = sm.OLS(labels, predictors)
    results = model.fit()
    # Print information
    print("-------------------------Processing Finshed %d---------------------------" % (i + 1))
    print(results.summary())
    print()
    
    # Check infomation for "#gohawks"
    if(i == 0):
    
        # Plot information         
        plt.gca().scatter(labels,predictors[:, 3],color = 'b')
        plt.title("Top 1 feature value: ranking score" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_1.png"
        plt.savefig(imageName)
        plt.gca().scatter(labels,predictors[:, 8],color = 'b')
        plt.title("Top 2 feature value: user ID" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_2.png"
        plt.savefig(imageName)
        plt.gca().scatter(labels,predictors[:, 5],color = 'b')
        plt.title("Top 3 feature value: mentions" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_3.png"
        plt.savefig(imageName)
        plt.close()

    # Check infomation for "#gohawks"
    if(i == 1):
    
        # Plot information         
        plt.gca().scatter(labels,predictors[:, 5],color = 'b')
        plt.title("Top 1 feature value: mentions" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_1.png"
        plt.savefig(imageName)
        plt.gca().scatter(labels,predictors[:, 3],color = 'b')
        plt.title("Top 2 feature value: ranking score" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_2.png"
        plt.savefig(imageName)
        plt.gca().scatter(labels,predictors[:, 1],color = 'b')
        plt.title("Top 3 feature value: followers" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_3.png"
        plt.savefig(imageName)
        plt.close()
        
    # Check infomation for "#gohawks"
    if(i == 2):
    
        # Plot information         
        plt.gca().scatter(labels,predictors[:, 5],color = 'b')
        plt.title("Top 1 feature value: mentions" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_1.png"
        plt.savefig(imageName)
        plt.gca().scatter(labels,predictors[:, 10],color = 'b')
        plt.title("Top 2 feature value: maximum followers" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_2.png"
        plt.savefig(imageName)
        plt.gca().scatter(labels,predictors[:, 9],color = 'b')
        plt.title("Top 3 feature value: tweets" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_3.png"
        plt.savefig(imageName)
        plt.close()
        
            # Check infomation for "#gohawks"
    if(i == 3):
    
        # Plot information         
        plt.gca().scatter(labels,predictors[:, 5],color = 'b')
        plt.title("Top 1 feature value: mentions" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_1.png"
        plt.savefig(imageName)
        plt.gca().scatter(labels,predictors[:, 1],color = 'b')
        plt.title("Top 2 feature value: followers" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_2.png"
        plt.savefig(imageName)
        plt.gca().scatter(labels,predictors[:, 3],color = 'b')
        plt.title("Top 3 feature value: ranking score" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_3.png"
        plt.savefig(imageName)
        plt.close()       
    # Check infomation for "#gohawks"
    if(i == 4):
    
        # Plot information         
        plt.gca().scatter(labels,predictors[:, 5],color = 'b')
        plt.title("Top 1 feature value: mentions" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_1.png"
        plt.savefig(imageName)
        plt.gca().scatter(labels,predictors[:, 1],color = 'b')
        plt.title("Top 2 feature value: followers" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_2.png"
        plt.savefig(imageName)
        plt.gca().scatter(labels,predictors[:, 3],color = 'b')
        plt.title("Top 3 feature value: ranking score" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_3.png"
        plt.savefig(imageName)
        plt.close()
        
    # Check infomation for "#gohawks"
    if(i == 5):
    
        # Plot information         
        plt.gca().scatter(labels,predictors[:, 4],color = 'b')
        plt.title("Top 1 feature value: maximum favorites" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_1.png"
        plt.savefig(imageName)
        plt.gca().scatter(labels,predictors[:, 2],color = 'b')
        plt.title("Top 2 feature value: retweets" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_2.png"
        plt.savefig(imageName)
        plt.gca().scatter(labels,predictors[:, 9],color = 'b')
        plt.title("Top 3 feature value: tweets" + "for " + hashtag_list[i])
        plt.xlabel('Feature value')
        plt.ylabel('Number of Tweets for Next Hour')
        plt.draw()
        imageName = hashtag_list[i] + "_top_feature_3.png"
        plt.savefig(imageName)
        plt.close()
