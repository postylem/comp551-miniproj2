# comp551-miniproj2
Group 27: Jacob Louis Hoover, Orla Mahon, Lambert De Monte

### Submitted Fri 18 Oct, 2019.

Our best classifier model is set up to run in the jupyter notebook `best-model.ipynb`. That notebook contains the scripts to process the data, and make the predictions, using the best-performing model pipeline that we could find (a Voting Classifier, called `voting_clf`, which is called by our gridsearch function, and will output the results and a confusion matrix plot). 

No external files are necessary to replicate the prediction we uploaded to Kaggle (except external python libraries, such as `pandas`, `numpy`, `nltk`, `matplotlib`, etc).  The predictions which we submitted with best accuracy on public leaderboard are also here: `predictions5798.csv`.

For our manually implemented Bernoulli Naive Bayes model, see the file `bernoulli_nb.ipynb`.  It requires `pandas` `numpy` and `scipy` packages.
