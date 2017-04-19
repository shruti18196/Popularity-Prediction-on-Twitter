"""
 EE 219 Project 5 Problem 7
 Name: Weikun Han
 Date: 3/22/2017
 Reference:
  - https://google.github.io/styleguide/pyguide.html
  - https://arxiv.org/abs/1401.2018
  - https://ucla.box.com/s/nv9td9kvvfvg3tya0dlvbs1kn5o87gmv
  - https://dev.twitter.com/docs
  - https://textblob.readthedocs.io/en/dev/
 Description:
  - Prepare Data Sets
  - City sentiment report
"""

from __future__ import print_function
from textblob import TextBlob
import json
import re
import matplotlib.pyplot as plt
import urllib
import datetime

class Reader:              
    def __init__(self):
        self.sentences = []
 
    def write_features(self, file_name):
        with open(file_name,'r') as reader :
            for line in reader :
                tweet_json = json.loads(line)
                
                tweet_date = tweet_json["firstpost_date"]
                post_time = datetime.datetime.fromtimestamp(tweet_date)


                if post_time.day == 01 and post_time.month == 02 and post_time.hour > 20:
                    sentence = tweet_json["tweet"]["text"]
                    loc = tweet_json["tweet"]["user"]["location"]               
                    if loc.lower().find("seattle") >= 0:
                        sentence = sentence.replace("@", "")
                        sentence = sentence.replace("\n", " ")
                        sentence = re.sub(r'[^\x00-\x7F]+',' ', sentence)
                        b = TextBlob(sentence)
                        if b.detect_language() != 'en': # Read next tweet
                            continue
                        
                        sentence_splits = sentence.split(" ")
                        final_words = []
                        for split in sentence_splits:
                            if split.startswith("http") or split.startswith("#") :
                                pass
                            else:
                                final_words.append(split)
                        final_string = ""
                        
                        for word in final_words:
                            final_string = final_string + " " + word
                        final_string.strip()
                        self.sentences.append(final_string)
                    
        print("-------------------------Processing Finshed 1----------------------")
        print("Total number of Strings read : %d" % len(self.sentences))
        print("--------------------------------------------------------------------") 
        print()
        
    def get_sentiments(self):
        f = open("seattle_sentiments.txt",'w')
        for i in range(0, len(self.sentences)):
            if self.sentences[i] == "":
                continue
            params = urllib.urlencode({'text': self.sentences[i]})
            file_url = urllib.urlopen("http://text-processing.com/api/sentiment/", params)
            data = file_url.read()
            try:
                response_json = json.loads(data)
                f.write(data + "\n")
            except :
                pass
        f.close()    

    def read_file(self):
        pos_total = 0
        pos_values = []
        neg_total = 0
        neg_values = []
        neutral_total = 0
        neutral_values = []
        values = []
        with open("seattle_sentiments.txt",'r') as reader :
            for line in reader:
                sentiment_json = json.loads(line)
                pos_value = sentiment_json["probability"]["pos"]
                neg_value = sentiment_json["probability"]["neg"]
                neutral_value = sentiment_json["probability"]["neutral"]
                pos_values.append(pos_value)
                neg_values.append(neg_value)
                neutral_values.append(neutral_value)                
                if pos_value > neg_value:
                    values.append(2)
                else:
                    values.append(1)
                pos_total += sentiment_json["probability"]["pos"]
                neg_total += sentiment_json["probability"]["neg"]
                neutral_total += sentiment_json["probability"]["neutral"]
        print("-------------------------Processing Finshed 2----------------------")
        print("Total Positive is: %d" % pos_total)
        print("Total Negative is: %d" % neg_total)
        print("Total Neutral is: %d" % neutral_total)
        print("--------------------------------------------------------------------") 
        print()
        plt.bar(range(0, len(values)), values, edgecolor='none',color='b')
        plt.xlabel("Number of Sentiment Word")
        plt.ylabel("Value of Sentiment Word")
        plt.title("Value of sentiment word per sentiment word in Seattle")
        plt.show()

# Print information
print(__doc__)
print()

r = Reader()    
r.write_features("tweet_data/tweets_#gohawks.txt")
r.get_sentiments()
r.read_file()
