import json
from textblob import TextBlob


# Reading in tweets in json format to be stored as json objects
json_data = open('data/TRAIN/NUS1.txt')
contents = json_data.readlines()
NUS1_tweets = []
for line in contents:
	NUS1_tweets.append(json.loads(line))

json_data = open('data/TRAIN/NUS2.txt')
contents = json_data.readlines()
NUS2_tweets = []
for line in contents:
	NUS2_tweets.append(json.loads(line))

json_data = open('data/TRAIN/DBS1.txt')
contents = json_data.readlines()
DBS1_tweets = []
for line in contents:
	DBS1_tweets.append(json.loads(line))

json_data = open('data/TRAIN/DBS2.txt')
contents = json_data.readlines()
DBS2_tweets = []
for line in contents:
	DBS2_tweets.append(json.loads(line))

json_data = open('data/TRAIN/STARHUB.txt')
contents = json_data.readlines()
STARHUB_tweets = []
for line in contents:
	STARHUB_tweets.append(json.loads(line))

print len(NUS1_tweets)
print len(NUS2_tweets)
print len(DBS1_tweets)
print len(DBS2_tweets)
print len(STARHUB_tweets)

"""
Steps to be done for each organization:

for each tweet in one organization
1) Tokenize into words
2) Implement spelling correction
3) Remove stop words (twitter stop word list)
4) Do stemming or lemmatization
5) Collate each term with their frequencies 
6) Assign weights to each term by calculating their tf-idf OR chi^2
7) Store the terms in a vector ranked by their weights (feature1)
8) Put the features for each tweet into the classifer to be trained
This shall be sufficient for a basic classifier.