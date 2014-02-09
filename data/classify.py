#!/usr/bin/env python
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
from textblob import TextBlob
import nltk
import codecs


def preprocess(text):
	# do spelling correction

	# remove punctuations and onvert to lowercase
	text = re.split(r'\W+', text.lower())

	# remove stopwords
	# text = [w for w in text if not w in stopwordlist]
	
	# stemming
	# stemmer = nltk.stem.porter.PorterStemmer()
	# text = [stemmer.stem(word) for word in text]

	return ' '.join(text)



# loading stopwords
stopwordlist = []
with open('stopwordlist.txt') as f:
	contents = f.readlines()
	for word in contents:
		stopwordlist.append(word.rstrip())	

orgs = ['DBS1', 'DBS2', 'NUS1', 'NUS2', 'STARHUB']
train_texts = []
train_orgs = []
test = [json.loads(line)['text'] for line in open('TEST/TEST_NEW.txt')]
groundtruths = [None] * len(test)
for org in orgs:
    with open('TRAIN/%s.txt' % org) as f:
        for line in f:
            train_texts.append(json.loads(line)['text'])
            train_orgs.append(org)
    with open('TEST/Groundtruth_%s.txt' % org) as f:
        for i, line in enumerate(f):
            if line == '1\n':
                groundtruths[i] = org
 
vectorizer = CountVectorizer(preprocessor=preprocess, stop_words=stopwordlist)
train_counts = vectorizer.fit_transform(train_texts)
classifier = MultinomialNB()
classifier.fit(train_counts, train_orgs)
 
test_counts = vectorizer.transform(test)
print classifier.score(test_counts, groundtruths)
