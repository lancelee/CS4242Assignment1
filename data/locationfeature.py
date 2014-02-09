import json
import re


orgs = ['DBS1', 'DBS2', 'NUS1', 'NUS2', 'STARHUB']
train_location = []
train_orgs = []
test = [json.loads(line)['text'] for line in open('TEST/TEST_NEW.txt')]
groundtruths = [None] * len(test)
for org in orgs:
    with open('TRAIN/%s.txt' % org) as f:
        for line in f:
            train_location.append(json.loads(line)['user']['location'])
            train_orgs.append(org)
    with open('TEST/Groundtruth_%s.txt' % org) as f:
        for i, line in enumerate(f):
            if line == '1\n':
                groundtruths[i] = org


print train_location                