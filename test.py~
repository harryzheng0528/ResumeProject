'''
#import Image
#from pytesseract import image_to_string
import json

#text = image_to_string(Image.open('Test_Resume-1.jpg'))
data = [] 
data.append({ 'id' : 1, 'text': 'Have you ever wanted to combine two or more dictionaries in Python. There are multiple ways to solve this problem. some are awkward, some are inaccurate, and most require multiple lines of code. Let walk through the different ways of solving this problem and discuss which is the most Pythonic. Our Problem Before we can discuss solutions, we need to clearly define our problem. Our code has two dictionaries: user and defaults. We want to merge these two dictionaries into a new dictionary called context.'}) 

data.append({ 'id' : 2, 'text': 'We have some requirements: user values should override defaults values in cases of duplicate keys keys in defaults and user may be any valid keys the values in defaults and user can be anything defaults and user should not change during the creation of context updates made to context should never alter defaults or user.'})

data.append({ 'id' : 3, 'text': 'hain items So far the most idiomatic way weve seen to perform this merge in a single line of code involves creating two lists of items, concatenating them, and forming a dictionary. We can join our items together more succinctly with itertools.chain: from itertools import chain This works well and may be more efficient than creating two unnecessary lists.'})

with open('data.json', 'w') as outfile:  
    json.dump(data, outfile)

'''
#necessary packages
from collections import defaultdict
from pyspark.mllib.linalg import Vector, Vectors
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.clustering import LDA, LDAModel
import json
import nltk

# define a dataframe and load data from HDFS
sc = SparkContext(appName="LDA_resume")
sqlContext = SQLContext(sc)
data = sqlContext.read.format("json").load("file:///home/hadoop_1/imagetotest/data.json")

# create a temp table "resume_text", use sql to get id and text information only
data.registerTempTable("resume_text")
text_resume = sqlContext.sql("select * from resume_text")
text_resume.show()
#print(text_resume.count())

#create a tokenizer
tokenizer = nltk.RegexpTokenizer(r'\w+')
en_stop = set(nltk.corpus.stopwords.words('english'))

#tokenize the sentence

token = text_resume.rdd \
		.map(lambda p: tokenizer.tokenize(p.text.lower())) \
		.map(lambda w: [x for x in w if x.isalpha()]) \
		.map(lambda w: [x for x in w if len(x) >= 3 and not x in en_stop])	

#wordcount for each word
termCount = token \
		.flatMap(lambda d: d) \
		.map(lambda w: (w,1)) \
		.reduceByKey(lambda x,y: x + y) \
		.map(lambda tuple: (tuple[1], tuple[0])) \
		.sortByKey(False)

# generate a vocabulary
vocabulary = termCount \
		.map(lambda x: x[1]) \
		.zipWithIndex() \
		.collectAsMap()
print(vocabulary)

#generate document word count vectors
def document_vector(document):
	id = document[0]
	counts = defaultdict(int)
	for token in document[1]:
		if token in vocabulary:
			token_id = vocabulary[token]
			counts[token_id] += 1
	counts = sorted(counts.items())
	keys = [x[0] for x in counts]
	values = [x[1] for x in counts]
	return (id, Vectors.sparse(len(vocabulary), keys, values))

documents = token.zipWithIndex().map(lambda index: (int(index[1]),index[0])) \
		.reduceByKey(lambda x, y: x+y) \
		.sortByKey(True) \
		.map(document_vector).map(list)
print(documents.take(3))

# Get an inverted vocabulary, so we can look up the word by it's index value
inv_voc = {value: key for (key, value) in vocabulary.items()}

#Train a LDA model
lda_model = LDA.train(documents, k=5, maxIterations=30)
topic_indices  = lda_model.describeTopics(maxTermsPerTopic=3)

#Print topics, showing the top-weighted 3 terms for each topic
for i in range(len(topic_indices)):
	print("Topic #{}\n".format(i+1))
	for j in range(len(topic_indices[i][0])):
		print("{}\t{}\n".format(inv_voc[topic_indices[i][0][j]].encode('utf-8'), topic_indices[i][1][j]))

#lda_model.save(sc,"file:///home/hadoop_1/imagetotest/LDAModel")
#sameModel = LDAModel.load(sc, "file:///home/hadoop_1/imagetotest/LDAModel")

y = lda_model.topicsMatrix()
print(type(y))
import numpy as np
new_document = documents.take(3)
x = []
for i in range(3):
	x.append(new_document[i][1].toArray())

print(type(x))
res = np.dot(x,y)
print(res)
	















