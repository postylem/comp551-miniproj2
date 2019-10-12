import numpy as np
import pandas as pd
import math
import time



def fit_naive_bayes():

    #Initialize marginal probability for each class
    marg_prob = np.array(20*[[0]])

    #Initialize matrix of probabilities of observed features given k
    num_features = 4
    cond_prob_matrix = np.empty((20,num_features))

    print(marg_prob)

    print(cond_prob_matrix)


def main():
    fit_naive_bayes()

if __name__ == "__main__":
    main()

    
