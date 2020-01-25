# --------------
# Importing Necessary libraries
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the 20newsgroups dataset
df = fetch_20newsgroups(subset='train')
pprint(list(df.target_names))

#Create a list of 4 newsgroup and fetch it using function fetch_20newsgroups
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',categories=categories)

#Use TfidfVectorizer on train data and find out the Number of Non-Zero components per sample.
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
print(vectors.shape)
print(vectors.nnz / float(vectors.shape[0]))

#Use TfidfVectorizer on test data and apply Naive Bayes model and calculate f1_score.
vectors_test = vectorizer.transform(newsgroups_test.data)
clf = MultinomialNB(alpha=.01)
clf.fit(vectors, newsgroups_train.target)
pred = clf.predict(vectors_test)
print("f1_score: ",f1_score(newsgroups_test.target, pred, average='macro'))

#Print the top 20 news category and top 20 words for every news category
import numpy as np
def show_top20(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top20 = np.argsort(classifier.coef_[i])[-20:]
        print("%s: %s" % (category, " ".join(feature_names[top20])))
show_top20(clf, vectorizer, newsgroups_train.target_names)


