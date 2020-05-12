# CS4035 Cyber Data Analytics
### Group 15

**Sharwin Bobde** (5011639)

**Nikhil Saldanha** (4998707)

## I. Data Pre-Processing

We prepare the data for visualization and modelling.

Functions used for data pre-processing can be found in [Part1/preprocessing.py](https://github.com/sharwinbobde/cyber-data-analytics/blob/master/Part-1/preprocessing.py)

### A. Re-Label

Column `simple_journal` has the payment status.

`relabel()`:
- drops rows which have `simple_journal` = "Refused", since we are not sure whether it was fraud.
- replaces `simple_journal` = "Chargeback" with `1`, i.e. fraud.
- replaces `simple_journal` = "Settled" with `0`, i.e. not fraud.

### B. Check for NaNs

**Remove NaNs from `issuercountrycode`, `shoppercountrycode` and `cardverificationcodesupplied`**

We replace NaN with `False` for `cardverificationcodesupplied` since we assume missing value means card verification code was not supplied.

Rest are replaced with dummy values which we numerically encode below

**Remove NaN in `mail_id`**

The NaNs in `mail_id` column are strings "na", which we replace with dummy value "email99999" and later encode it numerically

### C. Convert `creationdate` and `bookingdate` to a pandas datetime object

This allows more flexibility later for date wise aggregation using datetime functions

### D. Encode `issuecountrycode` and `shoppercountrycode`

- First we create a country code list that encapsulates both shopper and issue country codes
- Then we encode `issuecountrycode` and `shoppercountrycode` using this full list

### E. Convert transaction amount in `amount` to EUR

- The transaction amount in the `amount` column is in the currency according to the `currencycode` column
- We convert all transaction amounts to euros using a fixed exchange rate

### F. Replace `accountcode` with the respective Country Code

### G. Replace all `cvcresponsecode` values above 2 with a single value

`cvcresponsecode` 3 - 6 correspond to "not checked" and can be merged into one

### H. Numerical Encoding of categorical columns

We can replace the categorical columns with positive integers, since we need numbers to input to the ML Model

### I. Distribution of the Transaction Amount

- We analyse the distribution of `amount_eur` column for fraud and non-fraud cases
- We hope to find a significant difference between them which will allow us to classify the cases well
- To do this, we make a boxplot to see the quartiles and outliers

We can see here that there are outlier values for the non-fraud case. For further analysis, we remove the records where an amount > 180 is spent in a non-fraudulent transaction.

Now, if we compare this to the distribution of transaction amount in the fraudulent cases, there is a significant difference in distribution

## II. Visualization Task


### A. Heatmap with absolute difference in correlation

We usually visualize the correlation of the features with each other to see if there exist any meaningful relationships that differ in cases of fraud vs. no fraud

But it is really hard to see which parts are different for fraud vs. normal, so we create another heatmap with a difference in correlation so the important correlations are highlighted

We see clearly that many feature pairs with absolute difference between correlation for non-fraud and fraud in light blue blocks can discriminate one of the classes better than other. This shows that the interaction effect of some pairs of features have greater power of classifying either one of the classes. Here these are (amount, txid), (amount, cvcresponsecode), (cardverificationsupplied, shopperinterraction), etc.

---
### B. Visualize currencycode

Are transactions made in some currencies pre-disposed to having more fraudulent transactions?

---
### C. Visualize Card Type

Are transactions made using some credit card types pre-disposed to having more fraudulent transactions?

`txvariantcode` = 2 and 8 have a higher likelihood of being fraudulent

---

## D. Visualize Card Issuer

Do some transaction made using cards from certain card issuers entail more fraud?

---

### E. CVCResponseCode

Are transactions that have certain cvcresponse codes pre-disposed to being fraudulent?

We can see that the cvcresponsecode 0 is more prominent in cases of fraud, whereas 1 is more prominent in non-fraud cases

---

### F. Isomap

The isometrically consistent 2D representation shows us that the fraud cases are either outliers to the normal transaction patterns or form small clusters (shown by many overlapping red points in the plot) in the feature space. We can see this in the below visualization where some red dots appear outside the blue manifold and other cluster within

---

---

## III. Subsampling the Majority Class

The class imbalance between fraud and non-fraud is quite stark. We see the distribution below.

We would like to retain all fraud samples while sub-sampling about 10% of the non-fraud cases.

We do this below:

---

---

## IV. Feature Engineering and Standardization

Here, we create features by transforming and/or combining existing features.

### A. countries_equal: Shopped Country = Country in which Card was Issued

The intuition is that if the country in which the money was spent is not the same as the country in which the card was issued, it is likely that the transaction was fraudulent.

### B. Time of Transaction: Day of Week and Hour of Day

There is a clear difference in distribution between fraud and non-fraud. Fraudulent transactions tend to be distributed throughout the day, whereas non-fraudulent transactions gradually increase as the hour passes.

### BONUS TASK: Aggregate Features

We engineer the following features by aggregating over time for credit card. We use transaction aggregation strategy as a way to capture consumer spending behavior in the recent past. We put each credit card transaction into the historical context of past shopping behavior. In essence these attributes provide information on card holders' buying behavior in the immediate past.

- `prev_month_avg_amount`: Average amount spent in the previous month
- `prev_week_avg_amount`: Average amount spent in the previous week
- `prev_day_amount`: Average amount spent in the previous day
- `daily_avg_over_month`: Average amount spent per day in the last month of transactions
- `prev_day_same_country`: Number of transactions in the last 1 day in the same country as in this transaction
- `prev_month_same_country`: Number of transactions in the last 1 month in the same country as in this transaction
- `prev_month_avg_amount_same_country`: Average amount of transactions in the last month in the same country as in this transactions
- `prev_month_same_currency`: Number of transactions in the last 1 month with the same currency as in this transaction
- `prev_month_avg_amount_same_currency`: Average amount of transactions in the last month with the same currency as in this transactions
- `prev_total_transactions`: Total previous transactions with this credit card

Script used for creating aggregate features can be found in [Part1/aggregate_features.py](https://github.com/sharwinbobde/cyber-data-analytics/blob/master/Part-1/aggregate_features.py)

### NOTE: the cell below will take about 8 minutes

We visualize some of these below to show that they have different replationship with the label. These features are much lower and distributed more tightly around 0 for fraudulent transactions as compared to non-fraud transactions.

### Note: the above cell saves the data with all pre-processing to disk
### If you have run and saved this to disk once, run from the next cell to load data (and save time!)

---

---

### V. Classification Task

We decided to compare 2 models:
1. Decision Tree Classifier (White Box)
2. Support Vector Classifier (Black Box)

In our github repository:
- [Part-1/models.py](https://github.com/sharwinbobde/cyber-data-analytics/blob/master/Part-1/models.py) contains our models
- [Part-1/metrics.py](https://github.com/sharwinbobde/cyber-data-analytics/blob/master/Part-1/metrics.py) computes our metrics
- [Part-1/visualization.py](https://github.com/sharwinbobde/cyber-data-analytics/blob/master/Part-1/visualization.py) plots the AUC curve and feature importances

### Metrics

We evaluate the models based on:

1. The number of True Positives (higher is better)
2. The number of False Positives (lower is better)
3. The ROC curve

which shows how the model allows a tradeoff between TPs and FPs. We would like the model to have a high number of true positives while keeping the false positives low.

We also look at the Precision and Recall to justify which model should be used in which situation

### Standardize Data

We normalize the features by making the data zero mean and unit standard deviation. This is because most ML algorithms assume that the data is distributed normally and work well when data is distributed normally around 0.

### K Fold Cross Validation

We would like to evaluate our models on a train and validation set. We use the k fold cross validation method since it allows for multiple evaluations to get a variance estimate on our metrics

### Decision Tree Classifier (White Box Classifier)

A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes).

The paths from root to leaf represent classification rules. If a given sample and corresponding result is observable, the explanation for the result is easily explained by boolean logic. 
Making sure that we can explain why a transaction is labeled as being fraudulent.

Using the final decision tree with weighted notes, we can get feature importance.

**Feature importance** is calculated as the decrease in node impurity weighted by the probability of reaching that node. The node probability can be calculated by the number of samples that reach the node, divided by the total number of samples. The higher the value the more important the feature.

### Support Vector Classifier (Black-box Model)

A support-vector machine constructs a hyperplane or set of hyperplanes in a infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection. Intuitively, a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class - the margin, since in general the larger the margin, the lower the generalization error of the classifier.

The support vector machine method an be use to do non-linear classification as well by using the kernel trick. It often happens that the data are not linearly separable. Using the kernel trick, the original finite-dimensional space be mapped into a much higher-dimensional space, presumably making the separation easier in that space.

**This particular step of transferring a problem to a higher dim space makes SVC a black box model.**

### Result:

The SVC is able to get a higher task performance than the decision tree classifier in terms of AUC score (~ 90). This means that the model can very well distinguish between the two classes. The variance in the AUC is also quite small.

Additionally, the SVC correctly identifies 101 fraudulent cases while only reporting 2 False Positives. This is excellent news! Given that there are a many many more non-fraud cases, if the false positives were high, we would have to cancel credit cards of many people which would lead to inconvenience.

However, looking at the recall of the SVC, we see that it averages only about 0.29 across the folds which means that it can capture only about 30% of the total fraudulent transactions. Whereas, the decision tree model has a recall of over 50%. But it does this at the cost of a higher false positive rate (222 false positives).

A positive note about the decision tree is that it can explain its predictions through the feature importance and the boolean logic of how it classified a result. We can see the plot of the feature importances above. This can be helpful in many legal cases where a company might be required to tell a customer why it cancelled its transaction/credit card.

**In conclusion:** No one model is suitable for all cases

- if goal is explainability: choose decision tree (white box model)

- if false positives are expensive, eg: loss of customer due to incorrectly cancelling credit card, choose high precision model (in this case SVC)

- if false negatives are expensive, eg: letting go of some fraud cases, choose high recall model (in this case Decision Tree)

---

---

## VI. Imbalance Task
**by Student ID 4998707**

We use SMOTE to oversample the minority class to reduce the class imbalance. We have already undersampled the majority class earlier.

SMOTE is a technique to synthetically generate samples of the minority class so that they are similar to the existing samples in a vector space of their attributes. This is different from repeated oversampling which samples the minority class with replacement. This changes the overall distribution of the data and SMOTE is a way to retain a bit of the original distribution.

Below, we show our experiments. We train 3 models:

1. Decision Tree Classifier
2. Naive Bayes Classifier
3. Support Vector Classifier

on the SMOTEd and unSMOTEd data and compare the True/False Positives and corresponding AUCs.

For the unSMOTEd data, the class imbalance is 99.44% to 1.56%.. For the SMOTEd data, we restore the class balance to 50-50.

We were not able to train the SVC classifier on the SMOTEd data because the upsampling increase the dataset size too much that training takes forever.

Script used for SMOTE can be found in [Part-1/smote.py](https://github.com/sharwinbobde/cyber-data-analytics/blob/master/Part-1/smote.py)

### A. Decision Tree Classifier

### unSMOTEd

### SMOTEd data

---

---

### B. NaiveBayes Classifier

### unSMOTEd data

### SMOTEd data

---

---

### C. Support Vector Classifier

### unSMOTEd data

### SVC on SMOTEd data (did not run)

- does not complete in a reasonable amount of time
- requires a large amount of memory to store the covaraiance matrix
- hence infeasible to run on SMOTEd data

### Results

According to the results above, the support vector classifier performs the best among the three. It has an average AUC of around 90. It finds 101 positives accurately with only 2 false positives.

The NaiveBayes classifier is quite good as well since it gives an AUC of 86.

The Decision Tree Classifier does not work well after SMOTEing.

SMOTE is generally a good idea since it:
- allows us to increase the number of minority samples
- does not repeat the same samples while over sampling
- smartly oversample by interpolating between nearest neighbours to maintain the data distribution

SMOTE does not work in all cases, namely for simple Decision Trees since they are a bit different from other nonparametric statistical methods in that they cannot generalize to variations not seen in the training set. This is because a decision tree creates a partition of the input space and needs at least one example in each of the regions associated with a leaf to make a sensible prediction in that region. Since we do not do SMOTE for the validation/test set it performs poorly.

To be robust to these variations, one should use forests or even deeper architectures instead of trees, which provide a form of distributed representation and can generalize to variations not encountered in the training data

---

---

## VII. Rank Swapping 
**by Student ID 5011639**

We decided to apply rank swapping on all columns which may financial, personal or temporal information. This is because numeric as well as categorical feature may be used to link actual individuals to the records.

Lets see how the performance for NAive Bayes, Decision Tree and SVM are affected by rank-swapping by controlling the parameter `p` which controls the number of rows a partcular column value can be swapped within.

For the implementation of Rank Swapping please see [Part-1/rank_swapping.py](https://github.com/sharwinbobde/cyber-data-analytics/blob/master/Part-1/rank_swapping.py)

### Model AUC Plots for observing Rank Swapping effects.

### Results
The plots show that Rank Swapping has the following effect on the models:
*   **Naive Bayes**: ROC characteristics mostly unchanged but Precision-Recall curve sees major changes woth p=10 (more swapping) leading to greater generalization
*   **Decision Tree**: No change. Must be because it generalizes well as it is a rule based classifier
*   **Support Vector Machine**: minor changes in the ROC but no change in the Precision-Recall curve.

To answer the Rank Swapping question, with the intention of preserving users' privacy, rank-swapping does not come at a great cost of performance of the models. Thus, it is advisable to protect people's privacy using rank-swapping.