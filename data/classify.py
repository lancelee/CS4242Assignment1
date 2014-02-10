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
train_location = []
train_timezone = []
train_orgs = []
test_texts = []
test_location = []
test_timezone = []
for line in open('TEST/TEST_NEW.txt'):
    try:
        test_texts.append(json.loads(line)['text'])
    except:
        test_texts.append(None)
    try:
        test_location.append(json.loads(line)['user']['location'])
    except:
        test_location.append(None)
    try:
        test_timezone.append(json.loads(line)['user']['time_zone'])
    except:
        test_timezone.append(None)            
groundtruths = [None] * len(test_texts)
for org in orgs:
    with open('TRAIN/%s.txt' % org) as f:
        for line in f:
            try:
                train_texts.append(json.loads(line)['text'])                                
            except: 
                train_texts.append(None)
            try:
                train_location.append(json.loads(line)["user"]["location"])
            except:    
                train_location.append(None)
            try:
                train_timezone.append(json.loads(line)["user"]["time_zone"])
            except:    
                train_timezone.append(None)
            train_orgs.append(org)
    with open('TEST/Groundtruth_%s.txt' % org) as f:
        for i, line in enumerate(f):
            if line == '1\n':
                groundtruths[i] = org

pipeline = Pipeline([
    ('vect', CountVectorizer(max_df=0.5, stop_words=stopwordlist)),
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
    'clf__class_weight': (None, 'auto'),
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
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    # training phase
    grid_search.fit(train_texts, train_orgs)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    # testing classifer with testset and groundtruths
    print("Best score with test set: %0.3f" % grid_search.score(test_texts, groundtruths))
    predicted = grid_search.predict(test_texts)
    print(metrics.classification_report(groundtruths, predicted))
    print(metrics.confusion_matrix(groundtruths, predicted))



    # outputting wrong results
    output = open("false_prediction.txt", "wb")

    false_counts = 0
    for x in range(len(groundtruths)):
        if not predicted[x] == groundtruths[x]:
            false_counts += 1
            content = test_texts[x].encode('ascii', 'ignore')
            output.write(content + "\r\n")
            output.write("Predicted: " + predicted[x] + " Groundtruth: " + groundtruths[x] + "\r\n\r\n")

    print "false counts = " + str(false_counts)  

    output.close()  

# vectorizer = TfidfVectorizer(stop_words=stopwordlist)
# train_counts = vectorizer.fit_transform(train_texts)
# classifier = LinearSVC()
# classifier.fit(train_counts, train_orgs)
#
# test_counts = vectorizer.transform(test)
# print classifier.score(test_counts, groundtruths)
