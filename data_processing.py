import numpy as np
import pandas as pd
from k_fold import *
from sklearn.feature_extraction.text import CountVectorizer

    
def main(dictionary, stop_word_list):
    
    
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


    i = 1
    #################################################################
    # SELECT PARTICULAR CATEGORY OF COMMENTS
    #################################################################
    df = df.loc[df['subreddits'] == i]


    #################################################################
    # OUTPUT A NEW CSV OUT OF DATAFRAME
    #################################################################
    name = list(dictionary.keys())[list(dictionary.values()).index(i)]+'.csv'
    df.to_csv(name, index=False, sep = ',')


    #################################################################
    # COUNT OCCURENCES OF EACH WORDS ACROSS ONE COLUMN OF DATAFRAME
    #################################################################
    vectorizer = CountVectorizer(stop_words = stop_word_list)
    vectors_train = vectorizer.fit_transform(df['comments'])
    # print(vectorizer.vocabulary_)
    # print(vectorizer.get_feature_names())
    print(vectors_train.toarray()) 

    # vectors_train = vectorizer.transform(df['comments'])
    # print(vectors_train.shape)









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

    stop_word_list = ["a", "about", "above", "across", "after", "afterwards", 
    "again", "all", "almost", "alone", "along", "already", "also",    
    "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and",
    "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "as",
    "at", "be", "became", "because", "become","becomes", "becoming", "been", "before", 
    "behind", "being", "beside", "besides", "between", "beyond", "both", "but", "by","can", 
    "cannot", "cant", "could", "couldnt", "de", "describe", "do", "done", "each", "eg", 
    "either", "else", "enough", "etc", "even", "ever", "every", "everyone", "everything", 
    "everywhere", "except", "few", "find","for","found", "four", "from", "further", "get", 
    "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", 
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", 
    "however", "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least", 
    "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mine", "more", "moreover", 
    "most", "mostly", "much", "must", "my", "myself", "name", "namely", "neither", "never", 
    "nevertheless", "next","no", "nobody", "none", "noone", "nor", "not", "nothing", "now", 
    "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", 
    "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part","perhaps", 
    "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", 
    "should","since", "sincere","so", "some", "somehow", "someone", "something", "sometime", 
    "sometimes", "somewhere", "still", "such", "take","than", "that", "the", "their", "them", 
    "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", 
    "thereupon", "these", "they",
    "this", "those", "though", "through", "throughout",
    "thru", "thus", "to", "together", "too", "toward", "towards",
    "under", "until", "up", "upon", "us",
    "very", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", 
    "who", "whoever", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
    ]

    main(classes, stop_word_list)