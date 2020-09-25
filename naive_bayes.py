#Kevin Chawla
#1001543244
#kxc3244

#A program to create a naive bayes classifier using training and test data
import sys
import numpy as np
import pandas as pd


# Calculates mean and standard deviation for each attribute of each class
def stats(data):
    df = pd.DataFrame(data)
    df.rename(columns={len(data[0]) - 1: "class"}, inplace=True)
    averages = df.groupby(['class']).mean()
    std = df.groupby(['class']).std()
    groups = df.groupby(['class']).groups

    return averages, std, df, groups


# Function for Gaussian value using the formula provided
def gaussian(x, mean=0.0, sigma=1.0):
    x = float(x - mean) / sigma
    return np.exp(-x * x / 2.0) / np.sqrt(2.0 * np.pi) / sigma


# Calculate the probability of the class
def probability_class(data):
    class_set = {}
    class_probability = {}
    for i in range(len(data)):
        if data[i][len(data[i]) - 1] not in class_set:
            class_set[data[i][len(data[i]) - 1]] = 0
        else:
            class_set[data[i][len(data[i]) - 1]] += 1

    for i in class_set:
        class_probability[i] = class_set[i] / len(data)

    return class_probability


# Load training data and process it
def train(training_data):
    training = np.loadtxt(training_data)
    averages, stds, df, groups = stats(training)
    attribute = 0
    while True:
        try:
            currAttr = averages[attribute]
            attribute += 1
        except:
            break
    for i in groups.keys():
        for j in range(attribute):
            if stds[j][i] < 0.01:
                stds[j][i] = 0.01
            print("Class " + str(i) + ", attribute " + str(j + 1) + ", mean = " + str(
                "%.2f" % averages[j][i]) + ", std = " + str("%.2f" % stds[j][i]))

    class_probability = probability_class(training)

    return averages, stds, df, class_probability


# Load testing data and process it
def test(test_data, averages, stds, df, class_probability):
    test = np.loadtxt(test_data)
    test_attr = []
    for data_point in test:
        test_attr.append(data_point[:len(test[0]) - 1])

    predicted_values = []

    sorted_classes = sorted(class_probability)
    for i in range(len(test_attr)):
        gaussians = []
        for j in range(len(sorted_classes)):

            gaussians.append(1)
            for k in range(len(test_attr[i])):
                gaussians[j] *= gaussian(test_attr[i][k], averages[k][sorted_classes[j]], stds[k][sorted_classes[j]])
            gaussians[j] = gaussians[j] * class_probability[sorted_classes[j]]

        # Sum Rule to calculate P(x) and hence calculate the P(x|C) for each class
        gaussians[:] = [x / np.sum(gaussians) for x in gaussians]

        # maxim stores the max_values (more than one element if a tie exists)
        maxim = []
        for i in range(len(gaussians)):
            if len(maxim) == 0:
                maxim.append(gaussians[i])
            elif gaussians[i] == maxim[0]:
                maxim.append(gaussians[i])
            elif gaussians[i] > maxim[0]:
                maxim = []
                maxim.append(gaussians[i])

        # finds the index of probability class
        curr_set = {}
        for prob in maxim:
            curr_set[gaussians.index(prob) + 1]  = prob
        predicted_values.append(curr_set)
    accuracy = []

    for i in range(len(test)):
        predicted = 0
        key = list(predicted_values[i].keys())
        if (test[i][len(test[i]) - 1] in key):
            predicted = int(test[i][len(test[i]) - 1])
            accuracy.append(1 / len(key))
        else:
            predicted = int(key[0])
            accuracy.append(0)
        print("ID = " + str("%5d" % (i + 1)) + ", predicted = " + str("%3d" % predicted) + ", probabilty = " + str(
            "%.4f" % predicted_values[i][predicted]) + ", true = " + str(
            "%3d" % test[i][len(test[i]) - 1]) + ", accuracy = " + str("%4.2f" % accuracy[i]))
    classification_accuracy = np.mean(accuracy) * 100
    print("\nClassification Accuracy = " + "%6.4f" % classification_accuracy + "%")


def naive_bayes(training_file, test_file):
    averages, stds, df, class_probability = train(training_file)
    test(test_file, averages, stds, df, class_probability)


training_file = sys.argv[1]
test_file = sys.argv[2]

naive_bayes(training_file, test_file)
