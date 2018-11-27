from __future__ import division
import sys
import os.path
import numpy as np


import util
from util import DefaultDict
USAGE = "%s <test data folder> <spam folder> <ham folder>"

def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    word_to_count = DefaultDict(lambda : 0)
    for filename in file_list:
        unique_words_in_file = set()
        words_in_file = util.get_words_in_file(filename)
        for w in words_in_file:
            unique_words_in_file.add(w)
        for word in unique_words_in_file:
            word_to_count[word]+=1 
    return word_to_count  

def get_log_probabilities(file_list):
    """
    Computes log-probabilities for each word that occurs in the files in 
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    suma = len(file_list)
    word_to_count = get_counts(file_list)
    word_to_log_prob = DefaultDict(lambda : -np.log(suma+2))
    for key in word_to_count:
        word_to_log_prob[key] = (np.log(word_to_count[key]+1)-np.log(suma+2))


    return word_to_log_prob




def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files, 
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    spam_files, ham_files = file_lists_by_category
    n = len(spam_files)+len(ham_files)
  
    log_prior_by_category = []

    log_prior_by_category.append(np.log(len(spam_files) - np.log(n)))
    log_prior_by_category.append(np.log(len(ham_files) - np.log(n)))
   
    log_q = get_log_probabilities(spam_files)


    log_p = get_log_probabilities(ham_files)

    log_probabilities_by_category = [ log_q, log_p]

    return (log_probabilities_by_category, log_prior_by_category)




def classify_message(message_filename,
                     log_probabilities_by_category,
                     log_prior_by_category,
                     names = ['spam', 'ham']):
    """
    Uses Naive Bayes classification to classify the message in the given file.

    Inputs
    ------
    message_filename : name of the file containing the message to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    names : labels for each class (for this problem set, will always be just 
            spam and ham).

    Output
    ------


    One of the labels in names.
    """
    message_y = get_counts([message_filename]).keys()
    all_y = list(log_probabilities_by_category[0].keys())+\
    list(log_probabilities_by_category[1].keys())
    all_y  =set(all_y)

    log_q_spam = log_prior_by_category[0]
    for y in all_y:
        if y in message_y:
            log_q_spam += log_probabilities_by_category[0][y]
        else:
            log_q_spam+= np.log(1 - np.exp(\
                        log_probabilities_by_category[0][y]))


    log_p_ham = log_prior_by_category[1]

    for y in all_y:
        if y in message_y:
            log_p_ham += log_probabilities_by_category[1][y]
        else:
            log_p_ham+= np.log(1 - np.exp(\
                        log_probabilities_by_category[1][y]))

    if log_q_spam > log_p_ham:
        return "spam"
    else:
        return "ham"
 




if __name__ == '__main__':
    ### Read arguments
    if len(sys.argv) != 4:
        print(USAGE % sys.argv[0])
    testing_folder = sys.argv[1]
    (spam_folder, ham_folder) = sys.argv[2:4]

    ### Learn the distributions
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists)

    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_message(filename,
                                 log_probabilities_by_category,
                                 log_priors_by_category,
                                 ['spam', 'ham'])
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1


        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        #print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam messages, and %d out of %d ham messages."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))





