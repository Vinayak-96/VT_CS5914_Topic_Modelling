# CS 5914 - Topic Modeling
## Software Environment
- Python 3.9
- Packages
    - NLTK
    - Pandas
    - Beautiful Soup
    - PySpark
    - Gensim
    - spark-nlp
    - spaCy
    - pyLDAvis
    - Matplotlib
    - Numpy
    - pprint
    - re
- Set up HDFS
- Set up Spark
- Initialize environment variables (PYTHONPATH,PYSPARK_PYTHON)

To install spark-nlp, run this command "spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.4"

## Hardware Environment
The code was primarily run on a cluster of 3 Ubuntu 20.04 virtual machines with 96 GB storage.

# Running the scripts
- The VTechWorks Web Scraping.ipynb file should be run first. Either the Topic-Modelling-SparkNLP.py or the Gensim-TopicModeling.ipynb can be run after.
- Notebooks and Py Scripts can be run locally (make sure the csv location does not point to a HDFS location).

## VTechWorks Web Scraping:


#### How to run:
1. Import the libraries
2. Set the catalog variable in the 3rd cell to a VTechWorks catalog page's URL
3. Run the 3rd cell
4. Save the data to CSV with desired name in the 4th cell

## Topic Modelling SparkNLP:

#### How to run:
Just run the entire script. In the code, change the read.csv input to the location of the web scrapped file and to_csv() output to the your target location.

## Gensim Topic Modeling:

#### How to run:
Once all of the dependencies have been downloaded and imported, simply run through each cell at a time until reaching the fitting of the LDA Model (Note: the model assigned to variable "lda_model" is set with a specific initialization to ensure reproducability. If a random model is desired then run the cell with the variable set to "model"). 
Then, further cells are used for visualizations if desired.
