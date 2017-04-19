"""
 EE 219 Project 5 Problem 4
 Name: Weikun Han
 Date: 3/16/2017
 Reference:
  - https://google.github.io/styleguide/pyguide.html
  - https://arxiv.org/abs/1401.2018
  - https://ucla.box.com/s/nv9td9kvvfvg3tya0dlvbs1kn5o87gmv
  - https://dev.twitter.com/docs
  - http://statsmodels.sourceforge.net/
  - http://scikit-learn.org/stable/
 Description:
  - Linear Regression Model Using 10 Features
  - Ordinary Least Squares (OLS) Method
  - 10-Fold Cross-Validation with All Time Model
  - 10-Fold Cross-Validation with the First Period Time Model
  - 10-Fold Cross-Validation with the Second Period Time Model
  - 10-Fold Cross-Validation with the Third Period Time Model
"""

from __future__ import print_function
import os
import json
import datetime
from sets import Set
from datetime import timedelta
import os.path
import statsmodels.api as sm
from sklearn.cross_validation import KFold
import numpy as np


def retrieve_periodwise_hourly_features(file_id):
    st = datetime.datetime(2015,2,1,8,0,0) #Start time of the second period
    et = datetime.datetime(2015,2,1,20,0,0) #End time of the second period
    hourwise_features = {}
    users_per_hour = {} #Set to store the list of unique users
    file_name = os.path.join(folder_name, train_files[file_id])
    with open(file_name) as tweet_data: #Opening the respective file
        for individual_tweet in tweet_data: #For everyy line in the file
            individual_tweet_data = json.loads(individual_tweet) #Store an individual tweet as JSON object
            individual_time = individual_tweet_data["firstpost_date"] #The time when the tweet was posted
            individual_time = datetime.datetime.fromtimestamp(individual_time) 
            modified_time = datetime.datetime(individual_time.year, individual_time.month, individual_time.day, individual_time.hour, 0, 0)
            modified_time = unicode(modified_time)
            #Retrieving the user_id of the user who posted the tweet           
            individual_user_id = individual_tweet_data["tweet"]["user"]["id"] 
            #Retrieving the number of retweets          
            retweet_count = individual_tweet_data["metrics"]["citations"]["total"]
            #Retrieving the followers of the user           
            followers_count = individual_tweet_data["author"]["followers"]
            #Storing the total user user mentions in one tweet            
            user_mention_count = len(individual_tweet_data["tweet"]["entities"]["user_mentions"])
            #Storing the total urls in one tweet
            url_count = len(individual_tweet_data["tweet"]["entities"]["urls"])
            if url_count>0:
                url_count = 1 #If atleast 1 URL, then make it true, else false
            else:
                url_count = 0
            #Retrieving the list count of the user 
            listed_count = individual_tweet_data["tweet"]["user"]["listed_count"]
            if(listed_count == None):
                listed_count = 0
            #Obtaining the total number of 'likes' that a tweet has received            
            favorite_count = individual_tweet_data["tweet"]["favorite_count"]
            #Getting the rank of tweet            
            ranking_score = individual_tweet_data["metrics"]["ranking_score"]
            #Checking if the user is a verified user or not            
            user_verified = individual_tweet_data["tweet"]["user"]["verified"]
            #Inserting a new hour, initilizing features with zeros            
            if modified_time not in hourwise_features:
                hourwise_features[modified_time] = {'tweets_count':0, 'retweets_count':0, 'followers_count':0, 'max_followers':0, 'time':-1,'avg_user_mention_count':0,
                'url_count':0,'avg_listed_count':0,'max_listed_count':0,'avg_favorite_count':0,'max_favorite_count':0,'sum_ranking_score':0,'total_verified_users':0,'user_count':0}
                users_per_hour[modified_time] = Set([])
            hourwise_features[modified_time]['tweets_count'] += 1
            hourwise_features[modified_time]['retweets_count'] += retweet_count
            hourwise_features[modified_time]['time'] = individual_time.hour
            hourwise_features[modified_time]['avg_user_mention_count'] += user_mention_count
            hourwise_features[modified_time]['url_count'] += url_count
            hourwise_features[modified_time]['avg_favorite_count'] += favorite_count
            if favorite_count > hourwise_features[modified_time]['max_favorite_count']:
                hourwise_features[modified_time]['max_favorite_count'] = favorite_count
            hourwise_features[modified_time]['sum_ranking_score'] += ranking_score
            if individual_user_id not in users_per_hour[modified_time]:
                users_per_hour[modified_time].add(individual_user_id)
                hourwise_features[modified_time]['followers_count'] += followers_count
                hourwise_features[modified_time]['avg_listed_count'] += listed_count
                hourwise_features[modified_time]['user_count'] += 1
                if followers_count > hourwise_features[modified_time]['max_followers']:
                    hourwise_features[modified_time]['max_followers'] = followers_count
                if listed_count > hourwise_features[modified_time]['max_listed_count']:
                    hourwise_features[modified_time]['max_listed_count'] = listed_count
                if  (user_verified):
                    hourwise_features[modified_time]['total_verified_users'] += 1
    modified_hourwise_features = {}
    for time_value in hourwise_features:
        cur_hour = datetime.datetime.strptime(time_value, "%Y-%m-%d %H:%M:%S")
        features = hourwise_features[time_value]
        modified_hourwise_features[cur_hour] = features    
    all_keys = modified_hourwise_features.keys()
    period1_hourwise_features = {}   
    period2_hourwise_features = {}
    period3_hourwise_features = {}    
    for key in all_keys:
        if(key < st):
            period1_hourwise_features[key] = modified_hourwise_features[key]
        elif(key >= st and key <= et):
            period2_hourwise_features[key] = modified_hourwise_features[key]
        else:
            period3_hourwise_features[key] = modified_hourwise_features[key]
    return modified_hourwise_features, period1_hourwise_features, period2_hourwise_features, period3_hourwise_features
   

def variables_lables_matrix(hourwise_features):
    start_time = min(hourwise_features.keys()) #Find the start and end time from the data
    end_time = max(hourwise_features.keys())    
    predictors = []
    labels = []
    cur_hour = start_time
    while cur_hour <= end_time: #Keep looping from the start time till the end time
        next_hour_tweet_count = 0 #Initialize the label to be zero
        next_hour = cur_hour+timedelta(hours=1) #Go to the next hour
        if next_hour in hourwise_features:
            next_hour_tweet_count = hourwise_features[next_hour]['tweets_count'] #Update the label
        if cur_hour in hourwise_features:
            predictors.append(hourwise_features[cur_hour].values()) #Obtain the predictors
            labels.append([next_hour_tweet_count])
        else: #If a particular hour doesn't exist then initialize with zero
            temp = {'tweets_count':0, 'retweets_count':0, 'followers_count':0, 'max_followers':0, 'time':cur_hour.hour,'avg_user_mention_count':0,
                'url_count':0,'avg_listed_count':0,'max_listed_count':0,'avg_favorite_count':0,'max_favorite_count':0,'sum_ranking_score':0,'total_verified_users':0,'user_count':0}    
            predictors.append(temp.values())
            labels.append([next_hour_tweet_count])
        cur_hour = next_hour
    return predictors, labels

# Use cross-validation to combines (averages) measures of fit (prediction error)
def cross_validation(predictors,labels):
    all_prediction_errors = []
    
    # Use scikit-learn python API to do 10-fold cross-validation
    kf = KFold(len(predictors), n_folds=10)
    for train, test in kf:
        train_predictors = [predictors[i] for i in train]
        test_predictors = [predictors[i] for i in test]
        train_labels = [labels[i] for i in train]
        test_labels = [labels[i] for i in test]
        
        # Use Statsmodels python API to do linear regression models analysis
        train_labels = sm.add_constant(train_labels)        
        model = sm.OLS(train_labels, train_predictors)
        results = model.fit()
        test_labels_predicted = results.predict(test_predictors)
        prediction_error = abs(test_labels_predicted - test_labels)
        prediction_error = np.mean(prediction_error)
        all_prediction_errors.append(prediction_error)
    return np.mean(all_prediction_errors)        

# Print information
print(__doc__)
print() 
 
folder_name = "tweet_data"
train_files =  ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]
hashtag_list = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]

for i in range(len(train_files)):
    modified_hourwise_features, period1_hourwise_features, period2_hourwise_features, period3_hourwise_features = retrieve_periodwise_hourly_features(i)
    
    predictors, labels = variables_lables_matrix(modified_hourwise_features)
    average_cv_pred_error = cross_validation(predictors,labels)
    predictors1, labels1 = variables_lables_matrix(period1_hourwise_features)
    average_cv_pred_error1 = cross_validation(predictors1,labels1)
    predictors2, labels2 = variables_lables_matrix(period2_hourwise_features)
    average_cv_pred_error2 = cross_validation(predictors2,labels2)
    predictors3, labels3 = variables_lables_matrix(period3_hourwise_features)
    average_cv_pred_error3 = cross_validation(predictors3,labels3)
    
    # Print information
    print("-------------------------Processing Finshed %d---------------------------" % (i + 1))
    print("The all time average prediction error using 10-fold cross-validation for: [%s]" % hashtag_list[i])
    print("The average prediction error is: %0.3f" % average_cv_pred_error)
    print()
    print("The first period average prediction error using 10-fold cross-validation for: [%s]" % hashtag_list[i])
    print("The first period is before Feb. 1, 8:00 a.m.")
    print("The average prediction error is: %0.3f" % average_cv_pred_error1) 
    print()
    print("The second period average prediction error using 10-fold cross-validation for: [%s]" % hashtag_list[i])
    print("The second period is between Feb. 1, 8:00 a.m. and 8:00 p.m.")
    print("The average prediction error is: %0.3f" % average_cv_pred_error2) 
    print()
    print("The third period average prediction error using 10-fold cross-validation for: [%s]" % hashtag_list[i])
    print("The third period is after Feb. 1, 8:00 p.m.")
    print("The average prediction error is: %0.3f" % average_cv_pred_error3) 
    print("------------------------------------------------------------------------")
    print()
    
