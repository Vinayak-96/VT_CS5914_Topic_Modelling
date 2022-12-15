from pyspark.sql import functions as F
from pyspark.sql.functions import concat
from pyspark.sql import types as T
from pyspark.sql import SparkSession, Row

import sparknlp
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer
from sparknlp.annotator import Normalizer
from sparknlp.annotator import LemmatizerModel
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sparknlp.annotator import StopWordsCleaner
from sparknlp.annotator import NGramGenerator
from sparknlp.annotator import PerceptronModel
from sparknlp.base import Finisher

from pyspark.ml.feature import CountVectorizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import IDF
from pyspark.ml.clustering import LDA

import numpy as np
import matplotlib.pyplot as plt

#sparknlp can initialize a SparkSession
spark = sparknlp.start() 

#Reading the Data
data_path = 'hdfs://10.10.1.1:9000/user/vinayak/data/CS_papers.csv'
data = spark.read.csv(data_path, header=True)

#Topic Modelling Column
text_col = 'Abstract'
abstract_text = data.select(text_col).filter(F.col(text_col).isNotNull())


#Building spark-nlp steps for the pipeline
documentAssembler = DocumentAssembler() \
     .setInputCol(text_col) \
     .setOutputCol('document')
     
tokenizer = Tokenizer() \
     .setInputCols(['document']) \
     .setOutputCol('tokenized')
 
normalizer = Normalizer() \
     .setInputCols(['tokenized']) \
     .setOutputCol('normalized') \
     .setLowercase(True)

lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['normalized']) \
     .setOutputCol('lemmatized')

eng_stopwords = stopwords.words('english')



stopwords_cleaner = StopWordsCleaner() \
     .setInputCols(['lemmatized']) \
     .setOutputCol('unigrams') \
     .setStopWords(eng_stopwords)



ngrammer = NGramGenerator() \
    .setInputCols(['lemmatized']) \
    .setOutputCol('ngrams') \
    .setN(3) \
    .setEnableCumulative(True) \
    .setDelimiter('_')


pos_tagger = PerceptronModel.pretrained('pos_anc') \
    .setInputCols(['document', 'lemmatized']) \
    .setOutputCol('pos')



finisher = Finisher() \
     .setInputCols(['unigrams', 'ngrams', 'pos'])
     

#Build the spark-nlp pipeline
pipeline = Pipeline() \
     .setStages([documentAssembler,                  
                 tokenizer,
                 normalizer,                  
                 lemmatizer,                  
                 stopwords_cleaner, 
                 pos_tagger,
                 ngrammer,  
                 finisher])

#Apply it to the abstract text data
processed_text = pipeline.fit(abstract_text).transform(abstract_text)


#Combine unigrams and ngrams together in a column called "final"
processed_text = processed_text.withColumn('final', 
                                               concat(F.col('finished_unigrams'), 
                                                      F.col('finished_ngrams')))

#Fit Term Frequency vectorization using the "final" column
tfizer = CountVectorizer(inputCol='final', outputCol='tf_features')
tf_model = tfizer.fit(processed_text)
tf_result = tf_model.transform(processed_text)


#Use inverse document frequency to lower score for frequent words across all documents
idfizer = IDF(inputCol='tf_features', outputCol='tf_idf_features')
idf_model = idfizer.fit(tf_result)
tfidf_result = idf_model.transform(tf_result)


#Use the optimal value of topics. Check the other script with the gensim model to see how we obtain this value
#Unfortunately, spark-nlp does not provide a straight-forward method of obtaining the topic coherence score of the LDA model
num_topics = 12
max_iter = 100

#Fit the LDA model
lda = LDA(k=num_topics, maxIter=max_iter, featuresCol='tf_idf_features')
lda_model = lda.fit(tfidf_result)

transformed = lda_model.transform(tfidf_result)


#Visualize the word distribution for each topic
vocab = tf_model.vocabulary

def get_words(token_list):
     return [vocab[token_id] for token_id in token_list]
       
udf_to_words = F.udf(get_words, T.ArrayType(T.StringType()))

num_top_words = 10

topicsNew = lda_model.describeTopics(num_top_words).withColumn('topicWords', udf_to_words(F.col('termIndices')))
#Prints the topic number and word distribution for each topic
topicsNew.select('topic', 'topicWords').show(truncate=200)

#Save output to local file
results = topicsNew.select('topic', 'topicWords').toPandas()
results.to_csv("Spark_LDA_Results.csv", index = False)




#Visualization to obtain the number of documents for each topic in a bar graph
countTopDocs = transformed.select('topicDistribution')\
                .rdd.map(lambda r: Row( nTopTopic = int(np.argmax(r)))).toDF() \
                .groupBy("nTopTopic").count().sort("nTopTopic")

pdf = countTopDocs.toPandas()

pdf.plot(color = '#44D3A5', legend = False,
                           kind = 'bar', use_index = True, y = 'count', grid = False, figsize = (12, 10))
plt.xlabel('Topic')
plt.ylabel('Counts')
plt.title("Distribution of documents per topic")
plt.savefig("Spark_Basic_Distribution")