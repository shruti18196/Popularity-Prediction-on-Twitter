"""
 EE 219 Project 5 Problem 6 Part 1
 Name: Weikun Han
 Date: 3/16/2017
 Reference:
  - https://google.github.io/styleguide/pyguide.html
  - https://arxiv.org/abs/1401.2018
  - https://ucla.box.com/s/nv9td9kvvfvg3tya0dlvbs1kn5o87gmv
  - https://dev.twitter.com/docs
 Description:
  - Prepare Training and Test Sets
"""

from __future__ import print_function
import os
import json
import unicodedata

# Print information
print(__doc__)
print()    
    
# Determin which document you want to process
folder = "tweet_data"
dataset = "tweets_#superbowl.txt"
filename = os.path.join(folder, dataset)

# Setup the counter to determine number of document we want process
count = 0
count_wa = 0
count_ma = 0
total = 20000
subtotal = total * 0.6
massachusetts = ["Massachusetts", "MA", "Boston"]
washington = ["Washington", "WA", "Seattle"]

# Open one file in the dataset under the folder
with open(filename) as fp: 
        
    # Check each tweet in selected file in the dataset
    for line in fp:
        number1 = format(count_wa, "05d")
        number2 = format(count_ma, "05d")
        
        # Split 2 datasets into training(60%) and test(40%) sets, 
        filename1 = os.path.join("dataset_train", "washington", number1 + ".txt")
        filename2 = os.path.join("dataset_train", "massachusetts", number2 + ".txt")
        filename3 = os.path.join("dataset_test", "washington", number1 + ".txt")
        filename4 = os.path.join("dataset_test", "massachusetts", number2 + ".txt")
        tweet = json.loads(line) 
        location = tweet["tweet"]["user"]["location"]
        textual = tweet["tweet"]["text"]
        text = unicodedata.normalize("NFKD", textual).encode("ascii","ignore")
        if any(s in location for s in washington):
            if count_wa <= subtotal: 
                with open(filename1, "w") as file:
                    file.write(text + "\n")
                    #file.write("Washington\n")
                    #file.write("WA\n")
                    #file.write("Seattle")
                    file.close()
                    #print(location)
            if count_wa > subtotal and count_wa <= total: 
                with open(filename3, "w") as file:
                    file.write(text + "\n")
                    #file.write("Washington\n")
                    #file.write("WA\n")
                    #file.write("Seattle")
                    file.close()
                    print(location)
            count_wa = count_wa + 1
        if any(s in location for s in massachusetts):
            if count_ma <= subtotal: 
                with open(filename2, "w") as file:
                    file.write(text + "\n")
                    #file.write("Massachusetts\n")
                    #file.write("MA\n")
                    #file.write("Boston")
                    file.close()
                    #print(location)
            if count_ma > subtotal and count_ma <= total:
                with open(filename4, "w") as file:
                    file.write(text + "\n")
                    #file.write("Massachusetts\n")
                    #file.write("MA\n")
                    #file.write("Boston")
                    file.close()
                    #print(location)
            count_ma = count_ma + 1
        count = count + 1

# Print information    
print("-------------------------Processing Finshed 1----------------------")
print("The total tweets processed for hashtag [#superbowl] is: %d" % count)
print("The number of tweets posted by user located in Washington is: %d" % count_wa)
print("The number of tweets posted by user located in Massachusetts is: %d" % count_ma)
print("--------------------------------------------------------------------") 
print()

# Print information    
print("-------------------------Processing Finshed 2----------------------")
print("Prepared the number textual content of the tweet posted by user in Washington is: %d" % total)
print("The number of textual content in Washington in dataset_train is: %d" % subtotal)
print("The number of textual content in Washington in dataset_test is: %d" % (total - subtotal))
print("--------------------------------------------------------------------")  
print()  

# Print information    
print("-------------------------Processing Finshed 2----------------------")
print("Prepared the number textual content of the tweet posted by user in Massachusetts is: %d" % total)
print("The number of textual content in Massachusetts in dataset_train is: %d" % subtotal)
print("The number of textual content in Massachusetts in dataset_test is: %d" % (total - subtotal))
print("--------------------------------------------------------------------")  
print()  

