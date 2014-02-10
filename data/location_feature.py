import json
import re


def getLocationVector(train_location):
    feat = []
    # test
    for line in train_location:
        if not line is None:
            feat.append(line.lower())
    return feat



orgs = ['DBS1', 'DBS2', 'NUS1', 'NUS2', 'STARHUB']
train_location = []
train_orgs = []
test = [json.loads(line)['text'] for line in open('TEST/TEST_NEW.txt')]
groundtruths = [None] * len(test)


for org in orgs:
    with open('TRAIN/%s.txt' % org) as f:
        for line in f:
            try:
                location = json.loads(line)["user"]["time_zone"]
                if location == '':
                    try:
                        location = json.loads(line)["user"]["location"]
                        train_location.append(location)
                    except:
                        train_location.append(None)
                else:
                    train_location.append(location)            
            except:
                try:
                    location = json.loads(line)["user"]["location"]
                    train_location.append(location)
                except:
                    train_location.append(None)        
        	train_orgs.append(org)
    with open('TEST/Groundtruth_%s.txt' % org) as f:
        for i, line in enumerate(f):
            if line == '1\n':
                groundtruths[i] = org



