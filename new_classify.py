#!/usr/bin/env python
import json
from pprint import pprint
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
# from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import re
# from textblob import TextBlob
# import nltk
import codecs
import sys

import numpy as np
import matplotlib.pyplot as plt


"""
def preprocess(text):
    # do spelling correction

    # remove punctuations and onvert to lowercase
    # text = re.split(r'\W+', text.lower())

    # remove stopwords
    # text = [w for w in text if not w in stopwordlist]

    # stemming
    # stemmer = nltk.stem.porter.PorterStemmer()
    # text = [stemmer.stem(word) for word in text]

    return ' '.join(text)

def preprocessFeature(feature, train_texts):
    feat = []
    # test
    for x in range(len(train_texts)):
        if feature[x] == None:
            line = train_texts[x]
        else:
            line = preprocess(feature[x]) + " " + train_texts[x]
        feat.append(line.encode('ascii', errors='replace'))    
    return feat
"""

# loading stopwords
stopwordlist = []
with open('stopwordlist.txt') as f:
    contents = f.readlines()
    for word in contents:
        stopwordlist.append(word.rstrip())

orgs = ['DBS1', 'DBS2', 'NUS1', 'NUS2', 'STARHUB']
train_orgs = []

train_texts = []
test_texts = []

train_location = []
test_location = []

train_timezone = []
test_timezone = []

train_retweets = []
test_retweets = []

train_geoposition = []
test_geoposition = []

train_name = []
test_name = []

feat = []
testfeat = []

for line in open('TEST/TEST_NEW.txt'):
    json_dict = json.loads(line)
    text = json.loads(line)['text']
    try:
        location = json_dict["user"]["location"].replace(' ', '')
        if location:
            text += ' &' + location.lower()
            # print text
    except:
        pass

    try:
        time_zone = json_dict['user']['time_zone'].replace(' ', '')
        if time_zone:
            text += ' &' + time_zone.lower()
    except:
        pass

    try:
        geoposition = json_dict['geoposition'].replace(' ', '')
        if geoposition:
            text += ' &' + geoposition.lower()
    except:
        pass

    try:
        user_name = json_dict['user']['name'].replace(' ', '')
        if user_name:
            text += ' ' + user_name
    except:
        pass

    try:
        retweeted_status = json_dict['retweeted_status']['text']
        if retweeted_status:
            text += ' ' + retweeted_status
    except:
        pass

    test_texts.append(text)

main_groundtruths = []
for org in orgs:
    with open('TRAIN/%s.txt' % org) as f:
        for line in f:
            json_dict = json.loads(line)
            text = json.loads(line)['text']
            
            try:
                location = json_dict["user"]["location"].replace(' ', '')
                if location:
                    text += ' &' + location.lower()
            except:
                pass

            try:
                time_zone = json_dict['user']['time_zone'].replace(' ', '')
                if time_zone:
                    text += ' &' + time_zone.lower()
            except:
                pass

            try:
                geoposition = json_dict['geoposition'].replace(' ', '')
                if geoposition:
                    text += ' &' + geoposition.lower()
            except:
                pass
            
            try:
                user_name = json_dict['user']['name'].replace(' ', '')
                if user_name:
                    text += ' ' + user_name
            except:
                pass
            
            try:
                retweeted_status = json_dict['retweeted_status']['text']
                if retweeted_status:
                    text += ' ' + retweeted_status
            except:
                pass
            
            train_texts.append(text)

            train_orgs.append(org)
    
    groundtruths = [None] * len(test_texts)
    with open('TEST/Groundtruth_%s.txt' % org) as f:
        for i, line in enumerate(f):
            if line == '1\n':
                groundtruths[i] = 1
            elif line == '0\n':
                groundtruths[i] = 0
    main_groundtruths.append(groundtruths)

# generating training labels for each classifier
binary_train_orgs = []
DBS1_train_orgs = [None] * len(train_orgs)
DBS2_train_orgs = [None] * len(train_orgs)
NUS1_train_orgs = [None] * len(train_orgs)
NUS2_train_orgs = [None] * len(train_orgs)
STARHUB_train_orgs = [None] * len(train_orgs)

for i in range(len(train_orgs)):
    if train_orgs[i] == 'DBS1':
        DBS1_train_orgs[i] = 1
        DBS2_train_orgs[i] = 0
        NUS1_train_orgs[i] = 0
        NUS2_train_orgs[i] = 0
        STARHUB_train_orgs[i] = 0
    elif train_orgs[i] == 'DBS2':
        DBS1_train_orgs[i] = 0
        DBS2_train_orgs[i] = 1
        NUS1_train_orgs[i] = 0
        NUS2_train_orgs[i] = 0
        STARHUB_train_orgs[i] = 0
    elif train_orgs[i] == 'NUS1':
        DBS1_train_orgs[i] = 0
        DBS2_train_orgs[i] = 0
        NUS1_train_orgs[i] = 1
        NUS2_train_orgs[i] = 0
        STARHUB_train_orgs[i] = 0
    elif train_orgs[i] == 'NUS2':
        DBS1_train_orgs[i] = 0
        DBS2_train_orgs[i] = 0
        NUS1_train_orgs[i] = 0
        NUS2_train_orgs[i] = 1
        STARHUB_train_orgs[i] = 0
    elif train_orgs[i] == 'STARHUB':
        DBS1_train_orgs[i] = 0
        DBS2_train_orgs[i] = 0
        NUS1_train_orgs[i] = 0
        NUS2_train_orgs[i] = 0
        STARHUB_train_orgs[i] = 1
binary_train_orgs.append(DBS1_train_orgs)
binary_train_orgs.append(DBS2_train_orgs)
binary_train_orgs.append(NUS1_train_orgs)
binary_train_orgs.append(NUS2_train_orgs)
binary_train_orgs.append(STARHUB_train_orgs)



# comment these 2 lines to turn off location feature
#train_texts = preprocessFeature(train_location, train_texts)
#test_texts = preprocessFeature(test_location, test_texts)

#train_texts = preprocessFeature(train_timezone, train_texts)
#test_texts = preprocessFeature(test_timezone, test_texts)

#train_texts = preprocessFeature(train_retweets, train_texts)
#test_texts = preprocessFeature(test_retweets, test_texts)

#train_texts = preprocessFeature(train_geoposition, train_texts)
#test_texts = preprocessFeature(test_geoposition, test_texts)

#train_texts = preprocessFeature(train_name, train_texts)
#test_texts = preprocessFeature(test_name, test_texts)

def tokenizer(doc):
    # print doc

    # condense 3 or more than 3 letters into 1, e.g. hhhheeeello to hello
    # seems to decrease accuracy slightly
    # doc = re.compile(r'(\w)\1{2,}').sub(r'\1', doc)

    token_pattern = re.compile(r"(?u)[&\w]\w+")
    tokens = token_pattern.findall(doc)
    tokens = [token if token.lower() in ['dbs'] else token.lower() for token in tokens]
    # print tokens

    return tokens

pipeline = Pipeline([
    ('vect', CountVectorizer(max_df=0.5, stop_words=stopwordlist, lowercase=False, tokenizer=tokenizer)),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC()),
])

parameters = {
    # 'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'vect__stop_words': ('english', stopwordlist),
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    # 'clf__class_weight': (None, 'auto'),
    # 'clf__multi_class': ('ovr', 'crammer_singer'),
    # 'clf__C': (1.0, 2.0, 3.0, 4.0, 5.0),
    # 'clf__loss': ('l1', 'l2'),
    # 'clf__penalty': ('l1', 'l2'),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    binary_predicted = []
    for i in range(len(orgs)):
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

        print("Performing grid search for " + orgs[i] + " ...")
        # print("pipeline:", [name for name, _ in pipeline.steps])
        # print("parameters:")
        # pprint(parameters)
        t0 = time()
        # training phase
        grid_search.fit(train_texts, binary_train_orgs[i])
        print("done in %0.3fs" % (time() - t0))
        print("\n")

        print("Best score: %0.3f" % grid_search.best_score_)
        # print("Best parameters set:")
        # best_parameters = grid_search.best_estimator_.get_params()
        # for param_name in sorted(parameters.keys()):
        #    print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
        # testing classifer with testset and groundtruths
        print("Best score with test set: %0.3f" % grid_search.score(test_texts, main_groundtruths[i]))
        predicted = grid_search.predict(test_texts)
        print(metrics.classification_report(main_groundtruths[i], predicted))
        conf_matrix = metrics.confusion_matrix(main_groundtruths[i], predicted)
        print(conf_matrix)
        binary_predicted.append(predicted)
    


    
    output = open("binary.txt", "wb")

    false_counts = 0
    for x in range(len(binary_predicted[0])):
        result = ""
        for y in range(len(binary_predicted)):
            if binary_predicted[y][x] == 1:
                result += orgs[y] + " "
        output.write(result + "\r\n")

    output.close()
    

    
    # outputting wrong results
    output = open("new_false_prediction.txt", "wb")

    false_counts = 0
    for x in range(len(binary_predicted)):
        for y in range(len(binary_predicted[0])):
            if (binary_predicted[x][y] == 0) and (main_groundtruths[x][y] == 1):
                false_counts += 1
                content = test_texts[y].encode('ascii', 'ignore')
                output.write(content + "\r\n")
                output.write("This tweet should belong to " + orgs[x] + "\r\n\r\n")
    print "false counts = " + str(false_counts)  

    output.close()  
    
    print "Accuracy of results: " + str((1800 - false_counts) / 1800.0) + "\n"


# vectorizer = TfidfVectorizer(stop_words=stopwordlist)
# train_counts = vectorizer.fit_transform(train_texts)
# classifier = LinearSVC()
# classifier.fit(train_counts, train_orgs)
#
# test_counts = vectorizer.transform(test)
# print classifier.score(test_counts, groundtruths)