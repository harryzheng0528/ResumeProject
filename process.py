#necessary packages
from collections import defaultdict
from pyspark.mllib.linalg import Vector, Vectors
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.clustering import LDA, LDAModel
import json
import nltk
import numpy as np

# define a dataframe and load data from HDFS
sc = SparkContext(appName="LDA_resume")
sqlContext = SQLContext(sc)
data = sqlContext.read.format("json").load("file:///home/hadoop_1/ResumeProject/imagetotest/Data/data.json")

# create a temp table "resume_text", use sql to get id and text information only
data.registerTempTable("resume_text")
text_resume = sqlContext.sql("select id, text from resume_text")
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

new_document = documents.take(3)
x = []
for i in range(3):
	x.append(new_document[i][1].toArray())

print(type(x))
res = np.dot(x,y)/len(vocabulary)
print(res)
	
