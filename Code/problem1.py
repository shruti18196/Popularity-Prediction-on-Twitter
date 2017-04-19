"""
 EE 219 Project 5 Problem 1
 Name: Weikun Han
 Date: 3/16/2017
 Reference:
  - https://google.github.io/styleguide/pyguide.html
  - https://arxiv.org/abs/1401.2018
  - https://ucla.box.com/s/nv9td9kvvfvg3tya0dlvbs1kn5o87gmv
  - https://dev.twitter.com/docs
 Description:
  - Data Processing
"""

from __future__ import print_function
from sets import Set
import datetime
import os
import json
import matplotlib.pyplot as plt

# Use Twitter Developer Documentation python API to do calculation
def get_hashtag_statistics(file_id):

    # Initial information
    st = datetime.datetime(2017,03,17)
    et = datetime.datetime(2001,03,17) 
    unique_user_list = Set([]) 
    total_followers = 0.0
    total_retweets = 0.0
    total_tweets = 0.0
    total_hours = 0.0
    hour = {}
    file_name = os.path.join(folder, dataset[file_id]) 

    with open(file_name) as tweet_data: #Opening the respective file
        for individual_tweet in tweet_data: #For everyy line in the file
            individual_tweet_data = json.loads(individual_tweet) #Store an individual tweet as JSON object
            individual_user_id = individual_tweet_data["tweet"]["user"]["id"] #Retrieving the user_id of the user who posted the tweet
            if individual_user_id not in unique_user_list: #Add the user_id only if it is not in the list of all users and update the followers count
                total_followers += individual_tweet_data["author"]["followers"]
                unique_user_list.add(individual_user_id)
            total_retweets += individual_tweet_data["metrics"]["citations"]["total"] #Update the retweet count
            individual_time = individual_tweet_data["firstpost_date"] #The time when the tweet was posted
            individual_time = datetime.datetime.fromtimestamp(individual_time) 
            if individual_time < st:
                st = individual_time
            if individual_time > et:
                et = individual_time
            total_tweets = total_tweets + 1 #Update the number of tweets
            modified_time = datetime.datetime(individual_time.year, individual_time.month, individual_time.day, individual_time.hour, 0, 0)
            modified_time = unicode(modified_time)
            if modified_time not in hour:
                hour[modified_time] = {'hour_tweets_count':0}
            hour[modified_time]['hour_tweets_count'] += 1 
    total_hours = int((et - st).total_seconds()/3600 + 0.5) #Calculate the total hours from the time of first tweet till last tweet   
    return total_followers, total_retweets, total_tweets, total_hours, len(unique_user_list), hour

# Plot information 
def plot_histogram(file_id, hour):
    first_tweet_time = min(hour.keys())
    last_tweet_time = max(hour.keys())
    tweets_by_hour = []
    current_time = first_tweet_time
    while current_time <= last_tweet_time:
        if current_time in hour:
            tweets_by_hour.append(hour[current_time]["hour_tweets_count"])
        else:
            tweets_by_hour.append(0)
        current_time += datetime.timedelta(hours=1)        
    #Plotting the histogram
    plt.figure(figsize = (20, 10))
    plt.title("Number of Tweets per hour plot for " + hashtag_list[file_id])
    plt.ylabel("The Number of Tweets")
    plt.xlabel("The number of hours")
    plt.bar(range(len(tweets_by_hour)), tweets_by_hour)
    plt.show()

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
    hour_modified = {}
    total_followers, total_retweets, total_tweets, total_hours, total_users, hour = get_hashtag_statistics(i)
    
    # Print information
    print("-------------------------Processing Finshed %d---------------------------" % (i + 1))
    print("Average number of tweets per hour for [%s] is :%0.3f" % (hashtag_list[i], (total_tweets / total_hours)))
    print()
    print("Average number of followers of users posting tweets for [%s] is: %0.3f" % (hashtag_list[i], (total_followers / total_users)))
    print()
    print("Average number of retweets for [%s] is : %0.3f" % (hashtag_list[i], (total_retweets / total_tweets)))
    print("------------------------------------------------------------------------")
    print()
    
    for j in hour:
        cur_hour = datetime.datetime.strptime(j, "%Y-%m-%d %H:%M:%S")
        features = hour[j]
        hour_modified[cur_hour] = features    
    
    # Plot imformation
    if(i == 2 or i == 5):
        plot_histogram(i, hour_modified)
