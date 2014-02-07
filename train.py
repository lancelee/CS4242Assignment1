import json

from pprint import pprint
json_data = open('data/TRAIN/NUS1.txt')

contents = json_data.readlines()

list_of_tweets = []
for line in contents:
	list_of_tweets.append(json.loads(line))

json_data.close()

print len(list_of_tweets)