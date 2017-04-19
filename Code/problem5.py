"""
 EE 219 Project 5 Problem 5
 Name: Weikun Han
 Date: 3/17/2017
 Reference:
  - https://google.github.io/styleguide/pyguide.html
  - https://arxiv.org/abs/1401.2018
  - https://ucla.box.com/s/nv9td9kvvfvg3tya0dlvbs1kn5o87gmv
  - https://dev.twitter.com/docs
  - http://statsmodels.sourceforge.net/
  - https://ucla.box.com/s/ojvvthudugp9d2gze5nuep9ogwjydnur
 Description:
  - Linear Regression Model Using 10 Features
  - Ordinary Least Squares (OLS) Method
  - Another 10 Different Data Validation with the Corresponding Model
"""

from __future__ import print_function
import os
import json
import datetime
from sets import Set
from datetime import timedelta
import os.path
import statsmodels.api as sm
#import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
import numpy as np

'''
This function takes the entire hash tag file as the input and calculates all the features
'''


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

def retrieve_hourly_features(file_name):
    #st = datetime.datetime(2015,2,1,8,0,0) #Start time of the second period
    #et = datetime.datetime(2015,2,1,20,0,0) #End time of the second period
    hourwise_features = {}
    users_per_hour = {} #Set to store the list of unique users
    #file_name = os.path.join(folder_name, train_files[file_id])
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
    return modified_hourwise_features

 
'''
This function takes in the hourwise features for each hashtag as the input and computes the Predictor and Label Matrix
The labels are the tweet counts for the next hour
'''
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

# Print information
print(__doc__)
print() 
   
folder_name = "tweet_data"
train_files =  ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]
hashtag_list = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]
folder2 = "test_data"
dataset2 = ["sample1_period1.txt","sample2_period2.txt","sample3_period3.txt","sample4_period1.txt","sample5_period1.txt","sample6_period2.txt","sample7_period3.txt","sample8_period1.txt","sample9_period2.txt","sample10_period3.txt"]

x,period1_hourwise_features_sb, period2_hourwise_features_sb, period3_hourwise_features_sb = retrieve_periodwise_hourly_features(5)    
x,period1_hourwise_features_nfl, period2_hourwise_features_nfl, period3_hourwise_features_nfl = retrieve_periodwise_hourly_features(3)     
predictors1_sb, labels1_sb = variables_lables_matrix(period1_hourwise_features_sb)
predictors1_sb = sm.add_constant(predictors1_sb)      
predictors2_sb, labels2_sb = variables_lables_matrix(period2_hourwise_features_sb)
predictors2_sb = sm.add_constant(predictors2_sb)      
predictors3_sb, labels3_sb = variables_lables_matrix(period3_hourwise_features_sb)
predictors3_sb = sm.add_constant(predictors3_sb) 
predictors1_nfl, labels1_nfl = variables_lables_matrix(period1_hourwise_features_nfl)
predictors1_nfl = sm.add_constant(predictors1_nfl)      
predictors2_nfl, labels2_nfl = variables_lables_matrix(period2_hourwise_features_nfl)
predictors2_nfl = sm.add_constant(predictors2_nfl)      
predictors3_nfl, labels3_nfl = variables_lables_matrix(period3_hourwise_features_nfl)
predictors3_nfl = sm.add_constant(predictors3_nfl) 
model_1_sb = sm.OLS(labels1_sb, predictors1_sb)
results_1_sb = model_1_sb.fit()
model_2_sb = sm.OLS(labels2_sb, predictors2_sb)
results_2_sb = model_2_sb.fit()
model_3_sb = sm.OLS(labels3_sb, predictors3_sb)
results_3_sb = model_3_sb.fit()
model_1_nfl = sm.OLS(labels1_nfl, predictors1_nfl)
results_1_nfl = model_1_nfl.fit()
model_2_nfl = sm.OLS(labels2_nfl, predictors2_nfl)
results_2_nfl = model_2_nfl.fit()
model_3_nfl = sm.OLS(labels3_nfl, predictors3_nfl)
results_3_nfl = model_3_nfl.fit()

test_error = [0 for i in range(len(dataset2))]
for i in range(len(dataset2)):
    
    #The sample1_period1 file has maximum #superbowl tags, so we use the corresponding model
    if(i == 0):
        file_name = os.path.join("test_data", dataset2[i])
        modified_hourwise_features = retrieve_hourly_features(file_name)    
        predictors, labels = variables_lables_matrix(modified_hourwise_features)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)
        predictors = sm.add_constant(predictors)        
        test_labels_predicted_1 = results_1_sb.predict(predictors)
        
        # Should only use test_labels_predicted_1[0:4] - labels[1:5]
        test_error[i] = np.mean(abs(test_labels_predicted_1 - labels))
        
        # Print information
        print("-------------------------Processing Finshed %d---------------------------" % (i + 1))
        print("The sample1_period1 file have result as follows: ")
        print("If the next hour is 2, the predicted number of tweets is: %0.3f" % test_labels_predicted_1[0])
        print("If the next hour is 3, the predicted number of tweets is: %0.3f" % test_labels_predicted_1[1])
        print("If the next hour is 4, the predicted number of tweets is: %0.3f" % test_labels_predicted_1[2])
        print("If the next hour is 5, the predicted number of tweets is: %0.3f" % test_labels_predicted_1[3])
        print("If the next hour is 6, the predicted number of tweets is: %0.3f" % test_labels_predicted_1[4])
        print("If the next hour is 7, the predicted number of tweets is: %0.3f" % test_labels_predicted_1[5])
        print("The sample1_period1 file have the average prediction error is: %0.3f" % test_error[i])
        print("------------------------------------------------------------------------")
        print()
    
    #The sample2_period2 file has maximum #superbowl tags, so we use the corresponding model
    if(i == 1):
        file_name = os.path.join("test_data", dataset2[i])
        modified_hourwise_features = retrieve_hourly_features(file_name)    
        predictors, labels = variables_lables_matrix(modified_hourwise_features)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)
        predictors = sm.add_constant(predictors)        
        test_labels_predicted_2 = results_2_sb.predict(predictors)
        test_error[i] = np.mean(abs(test_labels_predicted_2 - labels))      
    
        # Print information
        print("-------------------------Processing Finshed %d---------------------------" % (i + 1))
        print("The sample2_period2 file have result as follows: ")
        print("If the next hour is 2, the predicted number of tweets is: %0.3f" % test_labels_predicted_2[0])
        print("If the next hour is 3, the predicted number of tweets is: %0.3f" % test_labels_predicted_2[1])
        print("If the next hour is 4, the predicted number of tweets is: %0.3f" % test_labels_predicted_2[2])
        print("If the next hour is 5, the predicted number of tweets is: %0.3f" % test_labels_predicted_2[3])
        print("If the next hour is 6, the predicted number of tweets is: %0.3f" % test_labels_predicted_2[4])
        print("If the next hour is 7, the predicted number of tweets is: %0.3f" % test_labels_predicted_2[5])
        print("The sample2_period2 file have the average prediction error is: %0.3f" % test_error[i])
        print("------------------------------------------------------------------------")
        print()  
           
    #The sample3_period3 file has maximum #superbowl tags, so we use the corresponding model        
    if(i==2):
        file_name = os.path.join("test_data", dataset2[i])
        modified_hourwise_features = retrieve_hourly_features(file_name)    
        predictors, labels = variables_lables_matrix(modified_hourwise_features)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)        
        predictors = sm.add_constant(predictors)        
        test_labels_predicted_3 = results_3_sb.predict(predictors)
        test_error[i] = np.mean(abs(test_labels_predicted_3 - labels)) 
        
        # Print information
        print("-------------------------Processing Finshed %d---------------------------" % (i + 1))
        print("The sample3_period3 file have result as follows: ")
        print("If the next hour is 2, the predicted number of tweets is: %0.3f" % test_labels_predicted_3[0])
        print("If the next hour is 3, the predicted number of tweets is: %0.3f" % test_labels_predicted_3[1])
        print("If the next hour is 4, the predicted number of tweets is: %0.3f" % test_labels_predicted_3[2])
        print("If the next hour is 5, the predicted number of tweets is: %0.3f" % test_labels_predicted_3[3])
        print("If the next hour is 6, the predicted number of tweets is: %0.3f" % test_labels_predicted_3[4])
        print("If the next hour is 7, the predicted number of tweets is: %0.3f" % test_labels_predicted_3[5])
        print("The sample3_period3 file have the average prediction error is: %0.3f" % test_error[i])
        print("------------------------------------------------------------------------")
        print() 
    
    #The sample4_period1 file has maximum #nfl tags, so we use the corresponding model
    if(i==3):
        file_name = os.path.join("test_data", dataset2[i])
        modified_hourwise_features = retrieve_hourly_features(file_name)    
        predictors, labels = variables_lables_matrix(modified_hourwise_features)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)        
        predictors = sm.add_constant(predictors)        
        test_labels_predicted_4 = results_1_nfl.predict(predictors)
        test_error[i] = np.mean(abs(test_labels_predicted_4 - labels))  
        
        # Print information
        print("-------------------------Processing Finshed %d---------------------------" % (i + 1))
        print("The sample4_period1 file have result as follows: ")
        print("If the next hour is 2, the predicted number of tweets is: %0.3f" % test_labels_predicted_4[0])
        print("If the next hour is 3, the predicted number of tweets is: %0.3f" % test_labels_predicted_4[1])
        print("If the next hour is 4, the predicted number of tweets is: %0.3f" % test_labels_predicted_4[2])
        print("If the next hour is 5, the predicted number of tweets is: %0.3f" % test_labels_predicted_4[3])
        print("If the next hour is 6, the predicted number of tweets is: %0.3f" % test_labels_predicted_4[4])
        print("If the next hour is 7, the predicted number of tweets is: %0.3f" % test_labels_predicted_4[5])
        print("The sample4_period1 file have the average prediction error is: %0.3f" % test_error[i])
        print("------------------------------------------------------------------------")
        print()        
    
    #The sample5_period1 file has maximum #nfl tags, so we use the corresponding model
    if(i==4):
        file_name = os.path.join("test_data", dataset2[i])
        modified_hourwise_features = retrieve_hourly_features(file_name)    
        predictors, labels = variables_lables_matrix(modified_hourwise_features)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)        
        predictors = sm.add_constant(predictors)        
        test_labels_predicted_5 = results_1_nfl.predict(predictors)
        test_error[i] = np.mean(abs(test_labels_predicted_5 - labels))         
    
        print("-------------------------Processing Finshed %d---------------------------" % (i + 1))
        print("The sample5_period1 file have result as follows: ")
        print("If the next hour is 2, the predicted number of tweets is: %0.3f" % test_labels_predicted_5[0])
        print("If the next hour is 3, the predicted number of tweets is: %0.3f" % test_labels_predicted_5[1])
        print("If the next hour is 4, the predicted number of tweets is: %0.3f" % test_labels_predicted_5[2])
        print("If the next hour is 5, the predicted number of tweets is: %0.3f" % test_labels_predicted_5[3])
        print("If the next hour is 6, the predicted number of tweets is: %0.3f" % test_labels_predicted_5[4])
        print("If the next hour is 7, the predicted number of tweets is: %0.3f" % test_labels_predicted_5[5])
        print("The sample5_period1 file have the average prediction error is: %0.3f" % test_error[i])
        print("------------------------------------------------------------------------")
        print()  
        
        
    #The sample6_period2 file has maximum #superbowl tags, so we use the corresponding model
    if(i==5):
        file_name = os.path.join("test_data", dataset2[i])
        modified_hourwise_features = retrieve_hourly_features(file_name)    
        predictors, labels = variables_lables_matrix(modified_hourwise_features)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)        
        predictors = sm.add_constant(predictors)        
        test_labels_predicted_6 = results_2_sb.predict(predictors)
        test_error[i] = np.mean(abs(test_labels_predicted_6 - labels))         
    
        print("-------------------------Processing Finshed %d---------------------------" % (i + 1))
        print("The sample6_period2 file have result as follows: ")
        print("If the next hour is 2, the predicted number of tweets is: %0.3f" % test_labels_predicted_6[0])
        print("If the next hour is 3, the predicted number of tweets is: %0.3f" % test_labels_predicted_6[1])
        print("If the next hour is 4, the predicted number of tweets is: %0.3f" % test_labels_predicted_6[2])
        print("If the next hour is 5, the predicted number of tweets is: %0.3f" % test_labels_predicted_6[3])
        print("If the next hour is 6, the predicted number of tweets is: %0.3f" % test_labels_predicted_6[4])
        print("If the next hour is 7, the predicted number of tweets is: %0.3f" % test_labels_predicted_6[5])
        print("The sample6_period2 file have the average prediction error is: %0.3f" % test_error[i])
        print("------------------------------------------------------------------------")
        print()  
    
    #The sample7_period3 file has maximum #nfl tags, so we use the corresponding model
    if(i==6):
        file_name = os.path.join("test_data", dataset2[i])
        modified_hourwise_features = retrieve_hourly_features(file_name)    
        predictors, labels = variables_lables_matrix(modified_hourwise_features)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)        
        predictors = sm.add_constant(predictors)              
        test_labels_predicted_7 = results_3_nfl.predict(predictors)
        test_error[i] = np.mean(abs(test_labels_predicted_7 - labels))         
    
        print("-------------------------Processing Finshed %d---------------------------" % (i + 1))
        print("The sample7_period3 file have result as follows: ")
        print("If the next hour is 2, the predicted number of tweets is: %0.3f" % test_labels_predicted_7[0])
        print("If the next hour is 3, the predicted number of tweets is: %0.3f" % test_labels_predicted_7[1])
        print("If the next hour is 4, the predicted number of tweets is: %0.3f" % test_labels_predicted_7[2])
        print("If the next hour is 5, the predicted number of tweets is: %0.3f" % test_labels_predicted_7[3])
        print("If the next hour is 6, the predicted number of tweets is: %0.3f" % test_labels_predicted_7[4])
        print("If the next hour is 7, the predicted number of tweets is: %0.3f" % test_labels_predicted_7[5])
        print("The sample7_period3 file have the average prediction error is: %0.3f" % test_error[i])
        print("------------------------------------------------------------------------")
        print() 
    
    #The sample8_period1 file has maximum #nfl tags, so we use the corresponding model
    if(i==7):
        file_name = os.path.join("test_data", dataset2[i])
        modified_hourwise_features = retrieve_hourly_features(file_name)    
        predictors, labels = variables_lables_matrix(modified_hourwise_features)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)        
        predictors = np.insert(predictors, [0], [[1],[1],[1],[1],[1]], axis=1)     
        test_labels_predicted_8 = results_2_sb.predict(predictors)
        test_error[i] = np.mean(abs(test_labels_predicted_8 - labels))    
        
        print("-------------------------Processing Finshed %d---------------------------" % (i + 1))
        print("The sample8_period1 file have result as follows: ")
        print("If the next hour is 2, the predicted number of tweets is: N/A")
        print("If the next hour is 3, the predicted number of tweets is: %0.3f" % test_labels_predicted_8[0])
        print("If the next hour is 4, the predicted number of tweets is: %0.3f" % test_labels_predicted_8[1])
        print("If the next hour is 5, the predicted number of tweets is: %0.3f" % test_labels_predicted_8[2])
        print("If the next hour is 6, the predicted number of tweets is: %0.3f" % test_labels_predicted_8[3])
        print("If the next hour is 7, the predicted number of tweets is: %0.3f" % test_labels_predicted_8[4])
        print("The sample8_period1 file have the average prediction error is: %0.3f" % test_error[i])
        print("------------------------------------------------------------------------")
        print()  

    #The sample9_period2 file has maximum #superbowl tags, so we use the corresponding model    
    if(i==8):
        file_name = os.path.join("test_data", dataset2[i])
        modified_hourwise_features = retrieve_hourly_features(file_name)    
        predictors, labels = variables_lables_matrix(modified_hourwise_features)
        predictors = sm.add_constant(predictors)        
        test_labels_predicted_9 = results_2_sb.predict(predictors)
        test_error[i] = np.mean(abs(test_labels_predicted_9 - labels)) 
    
        print("-------------------------Processing Finshed %d---------------------------" % (i + 1))
        print("The sample9_period2 file have result as follows: ")
        print("If the next hour is 2, the predicted number of tweets is: %0.3f" % test_labels_predicted_9[0])
        print("If the next hour is 3, the predicted number of tweets is: %0.3f" % test_labels_predicted_9[1])
        print("If the next hour is 4, the predicted number of tweets is: %0.3f" % test_labels_predicted_9[2])
        print("If the next hour is 5, the predicted number of tweets is: %0.3f" % test_labels_predicted_9[3])
        print("If the next hour is 6, the predicted number of tweets is: %0.3f" % test_labels_predicted_9[4])
        print("If the next hour is 7, the predicted number of tweets is: %0.3f" % test_labels_predicted_9[5])
        print("The sample9_period2 file have the average prediction error is: %0.3f" % test_error[i])
        print("------------------------------------------------------------------------")
        print()  
   
    #The sample10_period3 file has maximum #nfl tags, so we use the corresponding model     
    if(i==9):
        file_name = os.path.join("test_data", dataset2[i])
        modified_hourwise_features = retrieve_hourly_features(file_name)    
        predictors, labels = variables_lables_matrix(modified_hourwise_features)
        predictors = np.asarray(predictors)
        labels = np.asarray(labels)        
        predictors = np.insert(predictors, [0], [[1],[1],[1],[1],[1],[1]], axis=1)     
        test_labels_predicted_10 = results_3_nfl.predict(predictors)
        test_error[i] = np.mean(abs(test_labels_predicted_10 - labels)) 
        
        print("-------------------------Processing Finshed %d---------------------------" % (i + 1))
        print("The sample10_period3 file have result as follows: ")
        print("If the next hour is 2, the predicted number of tweets is: %0.3f" % test_labels_predicted_10[0])
        print("If the next hour is 3, the predicted number of tweets is: %0.3f" % test_labels_predicted_10[1])
        print("If the next hour is 4, the predicted number of tweets is: %0.3f" % test_labels_predicted_10[2])
        print("If the next hour is 5, the predicted number of tweets is: %0.3f" % test_labels_predicted_10[3])
        print("If the next hour is 6, the predicted number of tweets is: %0.3f" % test_labels_predicted_10[4])
        print("If the next hour is 7, the predicted number of tweets is: %0.3f" % test_labels_predicted_10[5])
        print("The sample10_period3 file have the average prediction error is: %0.3f" % test_error[i])
        print("------------------------------------------------------------------------")
        print() 
