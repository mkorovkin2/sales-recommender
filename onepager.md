# LinkedIn Sales Recommender

## Motivation
#### Problem
It is difficult to organically find sales peoplewith extensive amounts of experience
insales at software consulting firms. This task can require a lot of time to be spent
on LinkedIn, cold emails, and potentially high recruitment contracting costs.

#### Solution
Algorithm to classify an individual, basedon their previous work experience onLinkedIn,
as a candidate salesperson. For a given individual, the algorithm scrapes their work
history and details of their employers to determine whether: 1) they held a sales role,
and 2) whether they were employed by a software consulting firm. The results are combined
to make an overall prediction for the recruiting relevance of the individual.

The code provided uses two classifiers: one classifier to classify previous work history
and another classifier to classify employer companies as software consulting firms or not.
The scraper was made using Selenium and requires a user to log in before scraping.

## Research Summary
#### Classification With Random Forests
**Analysis summary**

The code written simulated the classification of individuals as being a good fit for a sales
role at a software consulting firm. The model was trained and tested on a labeled dataset
of individuals and companies. The classification was done using a RandomForest classifier
in Sci-Kit Learn. Work history classification was done based on work position titles and
company classification was done based on company descriptions (pulled from their "about"
section on LinkedIn). Text was vectorized using TfidfTransformer and CountVectorizer in
Sci-Kit Learn.  

The analysis additionally included the functionality of altering classification bounds. This
is because classifiers often output probabilities associated with predicted classes, so the
classification bound functionality takes advantage of this. For example, with a classification
bound of 0.55, any prediction probabilities lower than 0.55 will result in a negative class
and prediction probabilities higher than 0.55 will result in a a positive class. This allows
the user to more transparently control the model's classification process. Bounds can be
adjusted to any float that the user prefers (between 0 and 1). The bounds investigated
throughout the project were 0.50 to 0.70. 

**Dataset breakdown**
* Work history data: 11069 rows of work experience, for 2400+ individuals
* Company classifier: 970 companies

**Model Performance**

The model was trained using 80% of the entire dataset and validated using 20% of the
dataset. Performance results are displayed below. The statistics obtained below were
calculated based on the average results of 50 independent samples at a classification
bound of 0.56, which was found to be optimal for model performance.

* Work experience classifier (sales role or not)
    * Accuracy:             **94.48%**
    * True positive rate:   0.9034
    * False positive rate:  0.0966
    * True negative rate:   0.9658
    * False negative rate:  0.0342
    * F1 score:             0.9165
* Company classifier (software consulting or not)
    * Accuracy:             **89.76%**
    * True positive rate:   0.4040
    * False positive rate:  0.5960
    * True negative rate:   0.9553
    * False negative rate:  0.0447
    * F1 score:             0.4510
* Overall fit classifier (prior experience in sales at a software consulting firm)
    * Accuracy:             **92.88%**
    * True positive rate:   0.8423
    * False positive rate:  0.1577
    * True negative rate:   0.9854
    * False negative rate:  0.0146
    * F1 score:             0.9035

**Future Work**
* Automating the algorithmic process to avoid any human interaction
* Improving company classification accuracy
* Include role descriptions in the task of classifying work experience