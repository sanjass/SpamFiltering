import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import util

USAGE = "%s <test data folder> <spam folder> <ham folder>"
NUM_EXAMPLES = 100 # specify 0 or less to use all the training examples, or any positive integer to only train with that many
SHOW_LOSS_PLOT = True # whether or not to show the plot of loss decreasing over time

def extract_features(f, all_words):
    """
    Extract features from file for logistic regression. 

    Inputs
    ------
    f: Name of file to extract features from.

    all_words : List of all words in the training set of files. 

    Output
    ------
    Extracted features. 
    """
    words_in_file = util.get_words_in_file(f)
    words_in_file = set(words_in_file) 
    features = []
    for word in all_words:
        if word in words_in_file:
            features.append(1)
        else:
            features.append(0)
    return features

def new_extract_features(f, all_words):
    """
    part e) of pset
    Extract features from file for logistic regression. 

    Inputs
    ------
    f: Name of file to extract features from.

    all_words : List of all words in the training set of files. 

    Output
    ------
    Extracted features. Normalized frequency count 
    """
    words_in_file = util.get_words_in_file(f)
    # words_in_file = set(words_in_file) 
    features = np.zeros(len(all_words))
    for word in words_in_file:
        if word in all_words:
            features[all_words.index(word)]+=1
    return features/len(words_in_file)

def logistic_eval(y, c, theta):
    """
    Compute the cost function for a collection of data points.

    Note that this is just the sum of the cost for each individual data point.

    Inputs
    ------
    y: numpy array of features for the data, of shape (num_examples, num_features)
    
    c: numpy array of true class labels, either 0 or 1, of shape (num_examples)

    theta: numpy array of model parameters, theta_0 to theta_n, or shape (num_features + 1)

    Output
    ------
    logistic regression loss for model prediction compared to true class labels
    """
    cost = 0;
    for i in range(y.shape[0]):
        datapoint = y[i]
        
        sig = get_sigmoid(datapoint, theta)
        if c[i]==1:
            cost+= np.log(sig)
        else:
            cost+= np.log(1-sig)
    return cost


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def get_sigmoid(datapoint, theta):

  z = np.dot(datapoint, theta[1:])
  return sigmoid(z+theta[0])


def logistic_derivative(y, c, theta):
    """
    Compute the derivative of the cost function with respect to model parameters for a collection of data points.

    Note that this is just the sum of those derivatives for each data point individually. 

    Inputs
    ------
    y: numpy array of features for the data, of shape (num_examples, num_features)
    
    c: numpy array of true class labels, either 0 or 1, of shape (num_examples)

    theta: numpy array of model parameters, theta_0 to theta_n, or shape (num_features + 1)

    Output
    ------
    Numpy array G of shape (num_features + 1), where G[i] contains the derivative of the cost function
    with respect to theta[i]. 
    """
    G = np.zeros(y.shape[1]+1);
    for i in range(y.shape[0]):
        datapoint = y[i]
        sigmoid = get_sigmoid(datapoint, theta)
        G[1:]+= datapoint*(get_sigmoid(datapoint, theta)-c[i])
        G[0]+=get_sigmoid(datapoint,theta)-c[i]

           
    return G

def train_logistic(file_lists_by_category):
    """
    Extract features and labels for the logistic regression model, and train the model.

    Note that you'll need to arbitrarily pick one of spam and ham to be 0, and the other to be 1, 
    when creating the labels for logistic regression. 
    The choice doesn't matter; just make sure you are consistent about it. 

    Inputs
    ------
    A two-element list. The first element is a list of spam files, 
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    Tuple of (theta, all_words) where theta is the vector of trained logistic regression parameters, 
    and all_words is the list of all words found in the dataset (reused later to make
    sure we extract features in a consistent manner)
    """
    spam_files, ham_files = file_lists_by_category

    c = []
    features = []
    all_words = set()
    all_files = spam_files+ham_files
    for filename in all_files:
        words_in_file = util.get_words_in_file(filename)
        for word in words_in_file:
            all_words.add(word)

    all_words = list(all_words)
    for filename in spam_files:
        feature_y = extract_features(filename, all_words)
        c.append(1)
        features.append(feature_y)

    for filename in ham_files:
        feature_y = extract_features(filename, all_words)
        c.append(0)
        features.append(feature_y)



    features = np.array(features)
    labels = np.array(c)

    # We implemented optimize_theta for you later on in this file.
    # No need to spend a lot of time examining it, besides checking 
    # what it takes as input, unless you're curious. 
    # However, note that it relies on your previous functions to be correct.
    # Anyway, feel free to use e.g. the following code after you've created features and labels.

    theta = optimize_theta(features, labels, show_loss_plot=True)
    return theta, all_words

def classify_message(filename, theta, all_words):
    """
    Predict the label of a file, by just checking whether the logistic regression
    model outputs a prediction that is more or less than 0.5. 

    Inputs
    ------
    filename: Name of file to predict spam or ham on.

    theta: trained logistic regression parameters

    all_words: list of all words returned by train_logistic(). 

    Output
    ------
    'spam' or 'ham'
    """
    feature_y = []
   # words_in_file = util.get_words_in_file(filename)
    

    feature_y = extract_features(filename, all_words)

    prediction = get_sigmoid(feature_y, theta)
    if prediction>0.5:
        return "spam"
    else:
        return "ham"


"""

END OF YOUR CODE

DO NOT MODIFY ANY OF THE CODE THAT FOLLOWS.

"""



def optimize_theta(y, c, show_loss_plot=SHOW_LOSS_PLOT, learning_rate=0.5, convergence_threshold=1e-3):
    """
    Train the logistic regression model. 
    DO NOT MODIFY.

    Inputs
    ------
    y: numpy array of features for the data, of shape (num_examples, num_features)
    
    c: numpy array of true class labels, either 0 or 1, of shape (num_examples)

    show_loss_plot: whether or not to show the plot of logistic regression cost function over time.

    learning_rate: parameter controlling size of parameter update steps. the default should work fine.

    convergence_threshold: parameter controlling when we consider the model 'done training.' the default should work fine.

    Output
    -------
    Numpy array theta containing trained logistic model parameters theta_0, ... theta_n, of shape (num_features + 1)
    """
    num_examples, num_features = y.shape[0], y.shape[1]
    theta = np.array([0 for _ in range(num_features + 1)])

    prev_loss = float('inf')
    loss = -logistic_eval(y, c, theta)
    losses = [loss]
    #for _ in range(100):
    while loss < prev_loss - convergence_threshold * num_examples: # scale by num_examples as a normalization
        gradient = logistic_derivative(y, c, theta) / num_examples # scale by num_examples as a normalization
        theta = theta - learning_rate * gradient
        prev_loss = loss
        loss = -logistic_eval(y, c, theta)
        losses.append(loss)

    if show_loss_plot:
        plt.plot(range(len(losses)), losses)
        plt.xlabel('Gradient Descent Iterations')
        plt.ylabel('Logistic Regression Cost')
        plt.show()
    
    return theta

if __name__ == '__main__':
    ### Read arguments
    if len(sys.argv) != 4:
        print(USAGE % sys.argv[0])
    testing_folder = sys.argv[1]
    (spam_folder, ham_folder) = sys.argv[2:4]
    file_lists = []
    for folder in (spam_folder, ham_folder):
        if NUM_EXAMPLES > 0:
            file_lists.append(util.get_files_in_folder(folder)[:NUM_EXAMPLES])
        else:
            file_lists.append(util.get_files_in_folder(folder))
    print("Extracting Features and Training...")
    theta, all_words = train_logistic(file_lists)

    # # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    print("Testing...")
    idx = 1
    for filename in (util.get_files_in_folder(testing_folder)):
        idx += 1
        ## Classify
        label = classify_message(filename, theta, all_words)
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        # print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam messages, and %d out of %d ham messages."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))