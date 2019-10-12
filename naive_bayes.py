import numpy as np
import pandas as pd
import math
import time



def fit_naive_bayes(observations, y, num_features):

    #Initialize marginal probability for each class
    count_class = np.array(20*[[0]])
    marg_prob = np.array(20*[[1]]) #Laplace smoothing, starting counts with 1

    #Initialize matrix of probabilities of observed features given k
    cond_prob_matrix = np.empty((20,num_features))

    
    #compute marginal probability of each class
    total_comments = len(y)
    for i in range(20):
        for j in range(total_comments):
            if y[j][1] == i:
                count_class[i] += 1
    
    #Marginal probability for each class
    marg_prob = np.true_divide(count_class, total_comments)

    for i in range(len(observations)):
        feature_no = observations[i][0]
        comment_no = observations[i][1]
       
        comment_class = y[comment_no][1]
        cond_prob_matrix[comment_class][feature_no] += 1

    #divide each row of cond_prob_matrix by the count of comments per class
    for i in range(20):
        cond_prob_matrix[i] = np.true_divide(cond_prob_matrix[i], count_class[i])

    return marg_prob, cond_prob_matrix

def predict(observations, marg_prob, cond_prob_matrix):

    


def main():


    marginal, conditional = fit_naive_bayes(observations, y, num_features)




if __name__ == "__main__":
    main()

    
