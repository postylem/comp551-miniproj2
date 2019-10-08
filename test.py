#!/usr/bin/env python
# coding: utf-8

# In[50]:


# from https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568

import logging
import pandas as pd
import numpy as np
from numpy import random
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup


df = pd.read_csv('reddit-comment-classification-comp-551/reddit_train.csv')
df = df[pd.notnull(df['comments'])]
print(df.head(20))
print(df['comments'].apply(lambda x: len(x.split(' '))).sum())
plt.figure(figsize=(10,4))


df.subreddits.value_counts().plot(kind='bar');


# In[37]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def print_plot(index):
    example = df[df.index == index][['comments', 'subreddits']].values[0]
    if len(example) > 0:
        print(example[0])
        print('subreddit:', example[1])


def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text
    
df['comments'] = df['comments'].apply(clean_text)
df['subreddits'].apply(lambda x: len(x.split(' '))).sum()


# In[53]:


print_plot(1234)


# In[31]:


X = df.comments
y = df.subreddits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)


# In[43]:


values_array = np.unique(df.subreddits.values)


# In[ ]:





# In[ ]:





# In[ ]:





# In[51]:


from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

bernoulli_nb = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', BernoulliNB()),
                        ])
bernoulli_nb.fit(X_train, y_train)


multinomial_nb = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', BernoulliNB()),
                        ])
multinomial_nb.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred1 = bernoulli_nb.predict(X_test)
y_pred2 = multinomial_nb.predict(X_test)

print('bernoulli accuracy %s' % accuracy_score(y_pred1, y_test))
print(classification_report(y_test, y_pred1,target_names=values_array))

print('multinomial accuracy %s' % accuracy_score(y_pred2, y_test))
print(classification_report(y_test, y_pred2,target_names=values_array))


# In[ ]:




