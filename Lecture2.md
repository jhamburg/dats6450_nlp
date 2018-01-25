# Lecture 2 - Continuing with tf-idf
(https://nlp.stanford.edu/IR-book/html/htmledition/the-vector-space-model-for-scoring-1.html)

## Building a model with tf-idf

### A brief explanation of SVM
(https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/)

**What is Support Vector Machine?**

“Support Vector Machine” (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. However,  it is mostly used in classification problems. In this algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiate the two classes very well (look at the below snapshot).

![](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_1.png)

Support Vector Machine is a frontier which best segregates the two classes (hyper-plane/ line).

**How does it work?**
Above, we got accustomed to the process of segregating the two classes with a hyper-plane. Now the burning question is “How can we identify the right hyper-plane?”. Don’t worry, it’s not as hard as you think!

Identify the right hyper-plane (Scenario-1): Here, we have three hyper-planes (A, B and C). Now, identify the right hyper-plane to classify star and circle.

![](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_21.png)

You need to remember a thumb rule to identify the right hyper-plane: “Select the hyper-plane which segregates the two classes better”. In this scenario, hyper-plane “B” has excellently performed this job.

Identify the right hyper-plane (Scenario-2): Here, we have three hyper-planes (A, B and C) and all are segregating the classes well. Now, How can we identify the right hyper-plane?

![](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_3.png)

Here, maximizing the distances between nearest data point (either class) and hyper-plane will help us to decide the right hyper-plane. This distance is called as Margin. Let’s look at the below snapshot:

![](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_4.png)

Find the hyper-plane to segregate to classes (Scenario-3): In the scenario below, we can’t have linear hyper-plane between the two classes, so how does SVM classify these two classes? Till now, we have only looked at the linear hyper-plane.

![](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_8.png)

SVM solves this problem by introducing additional feature. Here, we will add a new feature z=x^2+y^2. Now, let’s plot the data points on axis x and z:

![](https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_9.png)

In above plot, points to consider are:

All values for z would be positive always because z is the squared sum of both x and y
In the original plot, red circles appear close to the origin of x and y axes, leading to lower value of z and star relatively away from the origin result to higher value of z.

In SVM, it is easy to have a linear hyper-plane between these two classes. But, another burning question which arises is, should we need to add this feature manually to have a hyper-plane. No, SVM has a technique called the *kernel trick. *

These are functions which takes low dimensional input space and transform it to a higher dimensional space i.e. it converts not separable problem to separable problem, these functions are called kernels. It is mostly useful in non-linear separation problem. Simply put, it does some extremely complex data transformations, then find out the process to separate the data based on the labels or outputs you’ve defined.

### Let's build a model!
(https://miguelmalvarez.com/2016/11/07/classifying-reuters-21578-collection-with-python/)


#### Core Concepts

Text classification  (a.k.a. text categorisation) is the task of assigning pre-defined categories to textual documents. This could be helpful to solve problems ranging from spam detection to language identification. Classification problems can be divided into different types according to the cardinality of the labels per document :

Binary: Only two categories exist and they are mutually exclusive. A document can either be in the category or not (e.g., Spam detection).
Multi-class: Multiple categories which are mutually exclusive (e.g., Language detection if we assume documents can only have one language)
Multi-label: Multiple categories with the possibility of multiple (or none) assignments (e.g., News categorisation, where a document could be about “Sports” and “Corruption” at the same time).
This list is not exhaustive (e.g., hierarchical classification, single class classification, …), but the majority of the problems fit one of these three traditional types of problems.

#### Representing Reuters

Here we transform the text, cleaning into a more useable format:
```
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
 
cachedStopWords = stopwords.words("english")
 
def tokenize(text):
  min_length = 3
  words = map(lambda word: word.lower(), word_tokenize(text))
  words = [word for word in words if word not in cachedStopWords]
  tokens = (list(map(lambda token: PorterStemmer().stem(token),
                                   words)))
  p = re.compile('[a-zA-Z]+');
  filtered_tokens =
    list(filter (lambda token: p.match(token) and
                               len(token) >= min_length,
                               tokens))
  return filtered_tokens
```

Reuters-21578 is arguably the most commonly used collection for text classification during the last two decade and it has been used in some of the most influential papers on the field. It contains structured information about newswire articles that can be assigned to several classes, making it a multi-label problem. It has a highly skewed distribution of documents over categories, where a large proportion of documents belong to few topics. 

The collection originally consisted of 21,578 documents but a subset and split is traditionally used. The most common split is Mod-Apte which only considers categories that have at least one document in the training set and the test set. The Mod-Apte split has 90 categories with a training set of 7769 documents and a test set of 3019 documents.

#### Classifying Reuters

In order to classify the collection, we have to apply a number of steps which are standard for the majority of classification problems:

- Define our training and testing subsets to make sure that we do not evaluate with documents that the system has learnt from. In our case, this is trivial as the original dataset is already split (for replicability purposes).
- Represent all the documents in each subset. Remember that any optimisation (e.g., IDF calculations) should be done in the training set only.
- Train a classifier on the represented training data.
- Predict the labels for each one of the represented testing documents.
- Compare the real and predicted document labels to evaluate our solution.
- We have chosen to use only one model (linear SVM) to simplify our solution. This model has traditionally produced good quality with text classification problems. Nonetheless, you should try multiple others models, as well as representations.

The problem we are solving has a multi-label nature, and because of this, there are two changes that we have to make in the code that are not needed for binary classification. 

1. Firstly, the data representation for the category assignment to the different documents is slightly different, viewing each document as a list of bits representing being or not in each of the categories. This change is done by using the MultiLabelBinarizer as the code shows. 

2. Secondly, we have to train our model (which is binary by nature) N times, once per category, where the negative cases will be the documents in all the other categories. This allows our model to make a binary decision per category and produce multi-label results. This can be done with the OneVsRestClassifier object in Scikit-learn. This step might change depending on the estimator you have chosen. For instance, some models (e.g., kNN) are multi-label by nature. You can find more info in the documentation.

```
from nltk.corpus import stopwords, reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
stop_words = stopwords.words("english")
 
# List of document ids
documents = reuters.fileids()
 
train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                            documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                           documents))
 
train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]
 
# Tokenisation
vectorizer = TfidfVectorizer(stop_words=stop_words,
                             tokenizer=tokenize)
 
# Learn and transform train documents
vectorised_train_documents = vectorizer.fit_transform(train_docs)
vectorised_test_documents = vectorizer.transform(test_docs)
 
# Transform multilabel labels
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id)
                                  for doc_id in train_docs_id])
test_labels = mlb.transform([reuters.categories(doc_id)
                             for doc_id in test_docs_id])
 
# Classifier
classifier = OneVsRestClassifier(LinearSVC(random_state=42))
classifier.fit(vectorised_train_documents, train_labels)
 
predictions = classifier.predict(vectorised_test_documents)
```

#### Evaluation

Measuring the quality of a classifier is a necessary step in order to potentially improve it. The main metrics for Text Classification are:

- Precision: Number of documents correctly assigned to a category out of the total number of documents predicted.
- Recall: Number of documents correctly assigned to a category out of the total number of documents in such category.
- F1: Metric that combines precision and recall using the harmonic mean.

If the evaluation is being done in multi-class or multi-label environments, the method becomes slightly more complicated because the quality metrics have to be either shown per category, or globally aggregated. There are two main aggregation approaches:

- Micro-average: Every assignment (document, label) has the same importance. Common categories have more effect over the aggregate quality than smaller ones.
- Macro-average: The quality for each category is calculated independently and their average is reported. All the categories are equally important.
Scikit-learn has functionality that will help us during this step as we can see below:

```
from sklearn.metrics import f1_score,
                            precision_score,
                            recall_score
 
precision = precision_score(test_labels, predictions,
                            average='micro')
recall = recall_score(test_labels, predictions,
                      average='micro')
f1 = f1_score(test_labels, predictions, average='micro')
 
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
        .format(precision, recall, f1))
 
precision = precision_score(test_labels, predictions,
                            average='macro')
recall = recall_score(test_labels, predictions,
                      average='macro')
f1 = f1_score(test_labels, predictions, average='macro')
 
print("Macro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
        .format(precision, recall, f1))
```        

This code shows that this baseline with the first model we tested and no optimisation whatsoever already produces reasonable quality levels with a micro-average F1 of 0.86 and a macro-average of 0.46.

Micro-average quality numbers
Precision: 0.9455, Recall: 0.8013, F1-measure: 0.8674
Macro-average quality numbers
Precision: 0.6493, Recall: 0.3948, F1-measure: 0.4665

## Clustering on text

## In class assignment: Cluster on social media!
1. Register for an app on twitter using these directions: (https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/)

2. Download the tweepy (pip install tweepy) module and pull tweets using this code: (https://gist.github.com/bonzanini/af0463b927433c73784d)

3. Then cluster using the example in the previous python notebook. 

4. Finally, characterize your clusters. What are the most important features for each cluster? Also extract a few representative tweets per cluster.
