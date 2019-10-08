import numpy as np
import pandas as pd
from k_fold import *
from sklearn.feature_extraction.text import CountVectorizer

    
def main(dictionary):
    
    
    #################################################################
    # CREATE K FOLDS OUT OF THE TRAINING SET
    #################################################################
    train_df = pd.read_csv("reddit-comment-classification-comp-551/reddit_train.csv")
    k = 5
    k_folds = k_fold(train_df, k)



    #################################################################
    # DROP ID COLUMN AND CHANGE NAME TO INT FOR THE CLASS COLUMN
    #################################################################
    df = k_folds[0].drop(['id'], axis=1)
    df['subreddits']= df['subreddits'].map(dictionary)


    #################################################################
    # SELECT PARTICULAR CATEGORY OF COMMENTS
    #################################################################
    df = df.loc[df['subreddits'] == 4]


    #################################################################
    # OUTPUT A NEW CSV OUT OF DATAFRAME
    #################################################################
    df.to_csv('canada.csv', index=False, sep = ',')


    #################################################################
    # COUNT OCCURENCES OF EACH WORDS ACROSS ONE COLUMN OF DATAFRAME
    #################################################################
    # vectorizer = CountVectorizer()
    # vectors_train = vectorizer.fit_transform(df['comments'])
    # print(vectorizer.vocabulary_)

if __name__ == "__main__":

    classes = {
        "anime": 1,
        "AskReddit": 2,
        "baseball": 3,
        "canada": 4, 
        "conspiracy": 5, 
        "europe": 6, 
        "funny": 7, 
        "gameofthrones": 8, 
        "GlobalOffensive": 9,
        "hockey" :10, 
        "leagueoflegends": 11, 
        "movies": 12, 
        "Music": 13, 
        "nba":14, 
        "nfl":15, 
        "Overwatch":16, 
        "soccer":17, 
        "trees":18, 
        "worldnews":19, 
        "wow":20
    }

    main(classes)