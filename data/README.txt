This dataset contains tweets from the following five organitions: 
NUS1(NATIONAL UNIVERSITY OF SINGAPORE)   (1000 training tweets)
NUS2(NATIONAL UNION OF STUDENTS)   (300 training tweets)
DBS1(DEVELOP BANK OF SINGAPORE)   (1000 training tweets)
DBS2(DEFENSIVE BACKS)   (200 training tweets)
STARHUB   (1000 training tweets)

The test file contains 1800 tweets. Besides, the groudtruth of these 1800 test tweets for each organization is also provided. 

The data is in json format, which contains all available information provided by Twitter.
For details about the defination of each field, please refer to
https://dev.twitter.com/docs/platform-objects/tweets
If you need to get more information (e.g., the social links between users), you could use the Twitter API:
https://dev.twitter.com/docs/api/1.1

A simple java program is provided to demonstrate how to read the json data, get specific data field, and generate word index.

A new file with several new tweets (in the same format with the test file) will be used to test your program during presentation. 
Please notice that the new tweets may contain words that are never shown in the training and testing dataset.

This dataset contains original data crowled from Twitter. 
Due to privacy issues, please do not public this dataset to anyone or for any use outside the class. Thank you.