# Loan Default Prediction

Prediction for severity of loan default given user information. Hackathon facilitated by [Analytics Vidhya](https://datahack.analyticsvidhya.com/contest/ltfs-datascience-finhack-an-online-hackathon/) and provided by Larsen Toubro Financial Services (LTFS). 

## Abstract
In this hackathon we deal with binary classification problem regarding the chance of default by borrower given accompanying information such as credit history, financial situation etc. We see how to deal with mixture of categorical and continuous data. The approach tries couple of models but only best is described here. 


## Approach
### Property of Data and Task:
1. Lot of categorical data.
2. Some continuous variable data as well
3. `Task:` Binary classification problem

### NaiveBayes (NB) Approach
1. NavieBayes has been very good practically for lot of complex task even though
assumption underneath are naive i.e. independence of observations.
2. Since most of the data are categorical where naive approach works well therfore
proceeded with this approach.
3. However, since the data is mix of categorical and continuous therefore vanillaNB will
not work well thus better to approach with splitting the features in two groups of categorical and
continuous.
4. Categorical features solved with [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) and
continuous group of featues solved with [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
5. Then output of both are combines in two ways:
    1. Combine the output and again use GaussisnNB
    2. Multply the probability of both the model output.
6. Multiplying the probability of output gives the best score, while for the sake of completeness
the score from only categorical and continuous was also tried but had inferior performance than combined
score of two NB models.
7. Missiing variable is handled in naive ways by replacing missing value with mean in case
of continuous variable and for categorical NaN is considered as a category for that feature.

### Conlusion
1. During the investigation with mixture of NB the normalized probability received by vanilla naive bayes trained on categorical and gaussian NB on continuous data seem to perform best and comparable with result obtained from gaussian NB bayes trained on output of vanilla naive bayes and gaussian naive bayes.     
2. Naive-Bayes algorithm is should at least work as baseline for modelling data which has lot of categorical data. In fact mix bo naive and gaussian bayes could serve as baseline for mix of categorical and continuous data. 
3. Other approaches which could be tried are greadient boosting and most importantly deep learning especially when you have lot of data. 


#### How to Use:
1. Use module <data_loader.py> to preprocess the data and store in pickle
2. Use module <sklearn_model.py> to train and predict. 