from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

# nltk.download('punkt')
import pandas as pd
import numpy as np
import sys, getopt

def main(argv):
    # Define default parameters
    dataframe_position_location = "example_position_dataframe.csv"
    dataframe_company_location = "example_company_dataframe.csv"
    number_of_samples_to_take = 50
    up_to_boundary = 20

    global df
    global df_companies

    # Get command line arguments
    # TODO: this
    #try:
    #    opts, args = getopt.getopt(argv, "h:p:o:n:m", ["position=", "company=", "samples=", "topbound="])
    #except getopt.GetoptError:
    #    'predictor.py -p [position dataframe] -c [company dataframe] -n [number of samples] -m [top boundary]'
    #    sys.exit(2)

    # Assemble dataframe of work histories
    df = pd.read_csv(dataframe_position_location)

    # Clean out all rows which contain non-unicode characters
    df['position'] = df["position"].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    df = df[df['position'].apply(lambda x: len(x.split(',')) < 4)]
    df = df.reset_index()

    # Assemble dataframe of companies
    df_companies = pd.read_csv(dataframe_company_location)

    # Build an easily-accessible set of company names for future use
    df_companies_set = set(list(df_companies.name))

    # Print statistics on the data
    print("- - - Starting - - -")
    print(f"Work history classification dataframe dimensions: {df.shape}")
    print(f"Company classification dataframe dimensions:      {df_companies.shape}")

    # By default, run simulation
    run_simulations(num_sample=number_of_samples_to_take, top_boundary=up_to_boundary)

def make_classifier_for_company(input_field, label_field):
    """Utility method that creates a classifier for companies as being a software consulting firm or not

    Parameters
    ----------
    input_field : str
        Column name in dataframe used to classify companies (default: 'description')
    print_cols : str
        Column nam used to label companies (default: 'label')

    Returns
    -------
    classifier : RandomForestClassifier
        Classifier used to classify companies
    count_vect : CountVectorizer
        Used for text vectorization
    X_test : pandas.DataFrame
        Dataframe of predictors used for validating model results
    y_test : numpy.array
        Array of labels used for validating model results

    """

    # Get rid of NA values
    df_companies[input_field].fillna("NA", inplace=True)

    # Split training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_companies[input_field], df_companies[label_field], test_size=0.2, random_state=0)

    # Vectorize words using Tfidf
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Make a classifier and return it
    nfclf = RandomForestClassifier()
    clf = nfclf.fit(X_train_tfidf, y_train)

    return clf, count_vect, X_test, y_test

def custom_train_test_split(df, input_field, label_field, test_prop):
    """Utility method to split up train and test sets in a custom fashion for individual classification

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of positions
    input_field : str
        Column name in dataframe used to classify companies (default: 'description')
    print_cols : str
        Column nam used to label companies (default: 'label')
    test_prop : float
        Proportion of dataframe to be used as test set

    Returns
    -------
    np.array
        Array of variables to be used as training set predictors
    np.array
        Array of variables to be used as test set predictors
    np.array
        Array of variables to be used as training set labels
    np.array
        Array of variables to be used as test set labels
    rettrain : list
        List of indices to be used as training set
    rettest : list
        List of indices to be used as test set

    """

    # Get dataframe size and proportions
    total_size = df.shape[0]
    test_size = np.int32(test_prop * total_size)
    sample_indices_test = np.random.choice(total_size, test_size, replace=False)

    # Calculate random test set locations
    full_range = set(np.arange(0, total_size))

    for i in sample_indices_test:
        if i in full_range:
            full_range.remove(i)

    # Combine results into return format
    rettrain = list(full_range)
    rettest = list(sample_indices_test)
    return np.array(df[input_field][rettrain]),\
           np.array(df[input_field][rettest]),\
           np.array(df[label_field][rettrain]),\
           np.array(df[label_field][rettest]),\
           rettrain, rettest

def make_classifier_for_person(input_field, label_field):
    """Utility method that creates a classifier for companies as being a software consulting firm or not

    Parameters
    ----------
    input_field : str
        Column name in dataframe used to classify companies (default: 'description')
    print_cols : str
        Column nam used to label companies (default: 'label')

    Returns
    -------
    classifier : RandomForestClassifier
        Classifier used to classify companies
    count_vect : CountVectorizer
        Used for text vectorization
    X_test : pandas.DataFrame
        Dataframe of predictors used for validating model results
    y_test : numpy.array
        Array of labels used for validating model results
    train_indices : list
        List of indices used for training set
    test_indices : list
        List of indices used for test set

    """

    # Get rid of NAs in dataframe
    df[input_field].fillna("NA", inplace=True)

    # Obtain custom train-test split
    X_train, X_test, y_train, y_test, train_indices, test_indicies = custom_train_test_split(df, input_field, label_field, 0.2)

    # Create word vectorization
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Build classifier
    nfclf = RandomForestClassifier()
    clf = nfclf.fit(X_train_tfidf, y_train)

    return clf, count_vect, X_test, y_test, train_indices, test_indicies

def classify_keyword(pos):
    """Method or positions based on keywords

    Parameters
    ----------
    pos : str
        Position title

    Returns
    -------
    float
        Total sum of keyword scores

    """

    # Define key set
    keyset = set([x.lower() for x in pos.split(" ")])
    word_dict = {
        "executive": 0.3,
        "account": 0.3,
        "accounts": 0.35,
        "sale": 0.5,
        "sales": 0.6,
        "recruiter": -0.4,
        "recruit": -0.4,
        "train": -0.3,
        "human": -0.1,
        "business": 0.1,
        "development": 0.2,
        "manager": 0.3,
        "director": 0.4,
        "global": 0.2,
        "representative": 0.2,
        "hr": -0.4,
        "software": -0.3,
        "developer": -0.1,
        "vp": 0.05,
        "leader": 0.1,
        "new": 0.1,
        "revenue": 0.2,
        "cro": 0.6,
        "ceo": 0.6,
        "specialist": 0.2,
        "market": 0.2
    }

    # Score the position title
    score = 0
    for k in keyset:
        if k in word_dict:
            score += word_dict[k]

    return score

def predict_person_label(ptest, clf, vect, clf_company, vect_company):
    """Method to classify an individual based on preior work experience and companies worked at

    Parameters
    ----------
    ptest : float
        Portion of dataset used for validation
    clf : RandomForestClassifier
        Classifier for individual's positions
    vect : CountVectorizer
        Count vectorizer for position
    clf_company : RandomForestClassifier
        Classifier for company
    vect_company : CountVectorizer
        Count vectorizer for company

    Returns
    -------
    names_dict : dict
        Dictionary of names and their classifications
    list
        Results of position classification statistics
    list
        Result of company classification statistics
    list
        Result of individual classification statistics

    """

    # Make sure to not classify duplicate people
    name_indices = df.name[ptest].unique()
    name_dict = {}

    # Performance statistics defined here
    ftp = 0
    ftn = 0
    ffp = 0
    ffn = 0
    ftt = 0

    person_true_positive = 0
    person_false_positive = 0
    person_true_negative = 0
    person_false_negative = 0
    person_total_count = 0

    company_true_positive = 0
    company_false_positive = 0
    company_true_negative = 0
    company_false_negative = 0
    company_total_count = 0

    ind_true_positive = 0
    ind_true_negative = 0
    ind_total_count = 0

    # Iterate through each person's name
    for name in name_indices:

        # Get a sub-dataframe of their work experience
        name_table = df[df['name'] == name]

        weights_array = []
        yes_probs = []
        company_results = []

        iteration = 0
        baseline_array = []

        # Define decision bounds
        dbound_company = 0.5
        dbound_person = 0.5

        rec_true_array = []

        # Make predictions for each position and company worked at
        for tab in name_table.iterrows():
            pos = tab[1]['position']
            rec = tab[1]['rec']
            com = tab[1]['company']
            com_desc_super = df_companies[df_companies['name'] == com]

            temp_company_true_positive = company_true_positive
            temp_company_false_positive = company_false_positive
            temp_company_true_negative = company_true_negative
            temp_company_false_negative = company_false_negative

            temp_person_true_positive = person_true_positive
            temp_person_false_positive = person_false_positive
            temp_person_true_negative = person_true_negative
            temp_person_false_negative = person_false_negative

            # Classify company if the description can be found
            if len(com_desc_super) > 0:
                com_desc = com_desc_super['description'].reset_index()['description'][0]
                com_label = com_desc_super['label'].reset_index()['label'][0]
                rec_true_array.append([rec, 1 if com_label == "Yes" else 0])

                result_c1 = clf_company.predict(vect_company.transform([com_desc]))
                result_c1p = clf_company.predict_proba(vect_company.transform([com_desc]))

                company_results.append(result_c1p[0][1])
                company_true_positive += 1 if (result_c1p[0][1] > dbound_company and rec == "Yes") else 0
                company_false_positive += 1 if (result_c1p[0][1] > dbound_company and rec == "No") else 0
                company_true_negative += 1 if (result_c1p[0][1] < dbound_company and rec == "No") else 0
                company_false_negative += 1 if (result_c1p[0][1] < dbound_company and rec == "Yes") else 0

            # If company cannot be classified, weight it as 0.5
            else:
                rec_true_array.append([rec, 0.5])

                company_results.append(0.5)
                company_true_negative += 1

            # Update counters
            company_total_count += 1

            # Obtain classification results for positions
            result_p1 = clf.predict_proba(vect.transform([pos]))

            # Update counters
            person_true_positive += 1 if (result_p1[0][1] > dbound_person and rec == "Yes") else 0
            person_false_positive += 1 if (result_p1[0][1] > dbound_person and rec == "No") else 0
            person_true_negative += 1 if (result_p1[0][1] < dbound_person and rec == "No") else 0
            person_false_negative += 1 if (result_p1[0][1] < dbound_person and rec == "Yes") else 0
            person_total_count += 1

            yes_probs.append(result_p1[0][1])

            # Build array of weighting
            weights_array.append(np.power(np.e, -1 / 5 * iteration))

            # Update baseline classification
            baseline_array.append(0.25)

            iteration += 1

            # Update company classification statistics
            if company_true_positive - temp_company_true_positive > 0 and person_true_positive - temp_person_true_positive > 0:
                ind_true_positive += 1

            elif company_true_negative - temp_company_true_negative > 0 and person_true_negative - temp_person_true_negative > 0:
                ind_true_negative += 1

            ind_total_count += 1

        # Obtain classification results
        total_result = 0
        baseline_result = 0
        sum_weight = np.sum(weights_array)

        real_result = 0

        # Obtain overall individual classification
        for i in range(len(weights_array)):
            weight = weights_array[i]
            yes = yes_probs[i]
            cm = company_results[i]
            baseline = baseline_array[i]

            total_result += weight * yes / sum_weight * cm
            baseline_result += weight * baseline / sum_weight * cm

            # Calcualte classification score
            true_rec = 1 if rec_true_array[i][0] == "Yes" else 0
            real_result += weight * true_rec / sum_weight * rec_true_array[i][1]

        # Classify individual based on baseline classification boundary
        temp_res = "Yes" if total_result > baseline_result else "No"
        name_dict[name] = temp_res
        real_res = "Yes" if real_result > baseline_result else "No"

        # Update classifier statistics
        if temp_res == real_res and real_res == "Yes":
            ftp += 1
        if temp_res == real_res and real_res == "No":
            ftn += 1
        if temp_res != real_res and real_res == "Yes":
            ffn += 1
        if temp_res != real_res and real_res == "No":
            ffp += 1
        ftt += 1

    return name_dict,\
           [person_true_positive, person_false_positive, person_true_negative, person_false_negative, person_total_count],\
           [company_true_positive, company_false_positive, company_true_negative, company_false_negative, company_total_count],\
           [ftp, ffp, ftn, ffn, ftt]

def simulate(score_bound=0.5,
             position=None,
             count_vect_position=None,
             company=None,
             count_vect_company=None,
             ptest=None):
    """Method to simulate classification of an individual

    Parameters
    ----------
    score_bound : float
        Probability boundary for a negative classificationb
    position : str
        Position title
    count_vect_position : CountVectorizer
        Count vectorizer for position
    company : RandomForestClassifier
        Classifier for company
    count_vect_company : CountVectorizer
        Count vectorizer for company
    ptest : float
        Proportion of datasets to be used as test set

    Returns
    -------
    names_dict : dict
        Dictionary of names and their classifications
    list
        Results of position classification statistics
    list
        Result of company classification statistics
    list
        Result of individual classification statistics

    """

    # If there is no position classification, get a position  and company classification & classifiers for them
    if position is None:
        position, count_vect_position, posx, posy, ptrain, ptest = make_classifier_for_person("position", "rec")
        company, count_vect_company, comx, comy = make_classifier_for_company("description", "label")

    # Get results by predicting the individual's label
    final_pred_classes, ptresults, ctresults, itresults = predict_person_label(ptest, position, count_vect_position, company, count_vect_company)

    return ptresults, ctresults, itresults, (position, count_vect_position), (company, count_vect_company), ptest

def output_test_stats(title, total_count, num_sample, std_list, class_bound, result_array):
    """Method to print out resulting model statistics

    Parameters
    ----------
    title : str
        A string to identify which classification the output statistics correspond
    total_count : int
        Total number of correct classifications
    num_sample : int
        Total number of individuals in sample
    std_list : list
        List of standard deviations
    class_bound : float
        Classification bound for negative classification
    result_array : list
        Classification result statistics (true positive, false positive, etc.)

    Returns
    -------
    Nothing

    """

    # Calculate mean
    mean = total_count / num_sample

    # Calculate standard deviation
    std = 0
    for s in std_list:
        std += np.power(s - mean, 2) / (num_sample - 1)

    # Print general outputs
    print(title + " accuracy:", str(np.round(mean, decimals=4)), "| std.dev =", np.round(np.sqrt(std), decimals=4), "| bound =",
          np.round(class_bound, decimals=2), "|", num_sample, "samples | dataset size =", df.shape[0])
    print("        [", np.round(mean - np.sqrt(std) * 1.976, decimals=4), ":",
          np.round(mean + np.sqrt(std) * 1.976, decimals=4), "] Confidence interval at 95%")

    if len(result_array) > 3:
        # Print sample-specific output
        print("        * True positive rate: ", np.round(result_array[0] / num_sample, decimals=4))
        print("        * False positive rate:", np.round(result_array[1] / num_sample, decimals=4))
        print("        * True negative rate: ", np.round(result_array[2] / num_sample, decimals=4))
        print("        * False negative rate:", np.round(result_array[3] / num_sample, decimals=4))
        print("        * F1 score:           ", np.round(result_array[4] / num_sample, decimals=4))

def manipulate_stats_array(arm):
    """Get statistical measures of classification

    Parameters
    ----------
    arm : list
        Input array of true positives, false positives, etc.

    Returns
    -------
    sub : list
        List of statistical measures of classification

    """

    sub = []

    # Avoid division by zero
    if arm[0] + arm[1] == 0:
        arm[0] = 1
    if arm[2] + arm[3] == 0:
        arm[2] = 1

    # True positives
    sub.append(arm[0] / (arm[0] + arm[1]))

    # False positives
    sub.append(arm[1] / (arm[0] + arm[1]))

    # True negatives
    sub.append(arm[2] / (arm[2] + arm[3]))

    # False negatives
    sub.append(arm[3] / (arm[2] + arm[3]))

    # Calculate F1 score
    precision = arm[0] / (arm[0] + arm[1])
    recall = arm[0] / (arm[0] + arm[3])
    sub.append(precision * recall * 2 / (precision + recall))

    return np.array(sub)

def run_simulations(num_sample=50, top_boundary=20):
    """Runs a given number of samples and outputs statistics of the sample of samples

    Parameters
    ----------
    num_sample : int
        Number of samples to be taken for each boundary comparison

    Returns
    -------
    Nothing

    """

    # Initialize variables to nothing
    position_ = None
    count_vect_position_ = None
    company_ = None
    count_vect_company_ = None
    ptest_ = None

    # Decimal boundaries to check - which boundary classification produces the highest accuracy
    # Example 1: boundary[0] = 5, suggests that the classification boundary to be checked is 0.5 + 0.05
    # Example 2: boundary[1] = 6, suggests that the classification boundary to be checked is 0.5 + 0.06
    # Example 3: boundary[2] = 7, suggests that the classification boundary to be checked is 0.5 + 0.07

    boundaries_to_check = range(top_boundary)

    for i in boundaries_to_check:

        # Classification boundary
        ix = 0.5 + i / 100

        print('Checking classification boundary ' + str(ix))

        # Initialize tracking statistics
        total_count_p = 0
        std_list_p = []
        parr = []

        total_count_c = 0
        std_list_c = []
        carr = []

        total_count_i = 0
        std_list_i = []
        iarr = []

        # Iterate through samples
        for sampnum in range(num_sample):

            # Output nice statistics
            print("Running sample #" + str(sampnum + 1) + "\t " + str(np.round(100 * sampnum / num_sample, decimals=2)) + "% complete")
            ptr, ctr, itr, p, c, ptestv = simulate(ix,
                                           position=position_,
                                           count_vect_position=count_vect_position_,
                                           company=company_,
                                           count_vect_company=count_vect_company_,
                                           ptest=ptest_)

            # Record last variables to avoid recomputing classifiers and count vectorizers
            if position_ is None:
                position_ = p[0]
                count_vect_position_ = p[1]
                company_ = c[0]
                count_vect_company_ = c[1]
                ptest_ = ptestv

            # Get statistics for position, company, and individual accuracy

            # Position
            accp = (ptr[0] + ptr[2]) / ptr[4]

            # Company
            accc = (ctr[0] + ctr[2]) / ctr[4]

            # Individual
            acci = (itr[0] + itr[2]) / itr[4]

            # For each item, make sure the results get tracked if it is returned
            if len(parr) < 1:
                parr = manipulate_stats_array(ptr)
            else:
                parr = parr + manipulate_stats_array(ptr)

            if len(carr) < 1:
                carr = manipulate_stats_array(ctr)
            else:
                carr = carr + manipulate_stats_array(ctr)

            if len(iarr) < 1:
                iarr = manipulate_stats_array(itr)
            else:
                iarr = iarr + manipulate_stats_array(itr)

            std_list_p.append(accp)
            total_count_p += accp

            std_list_c.append(accc)
            total_count_c += accc

            std_list_i.append(acci)
            total_count_i += acci

        # Print statistical findings nicely
        output_test_stats("Exp", total_count_p, num_sample, std_list_p, ix, parr)
        output_test_stats("Cmp", total_count_c, num_sample, std_list_c, ix, carr)
        output_test_stats("Ind", total_count_i, num_sample, std_list_i, ix, iarr)

if __name__ == "__main__":
   main(sys.argv[1:])