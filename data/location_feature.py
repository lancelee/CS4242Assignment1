import json
import re

orgs = ['DBS1', 'DBS2', 'NUS1', 'NUS2', 'STARHUB']
train_location = []
train_orgs = []
test = [json.loads(line)['text'] for line in open('TEST/TEST_NEW.txt')]
groundtruths = [None] * len(test)

output = open("timezone.txt", "wb")

for org in orgs:
    output.write("Data for " + org + '\r\n')
    with open('TRAIN/%s.txt' % org) as f:
        for line in f:
            try:
                train_location.append(json.loads(line)["user"]["time_zone"])
                output.write(json.loads(line)["user"]["time_zone"] + '\r\n')        	
            except:
                train_location.append(None)
                output.write("None" + '\r\n')
        	train_orgs.append(org)
    with open('TEST/Groundtruth_%s.txt' % org) as f:
        for i, line in enumerate(f):
            if line == '1\n':
                groundtruths[i] = org

output.close()