from math import log
from collections import Counter

def log2(x):
    return log(x,2)

def inner(x,y):
    return sum(xi*yi for (xi,yi) in zip(x,y))

def entropy(database,weights):
    #TODO: use weights


    negative_examples = [example[-1] == 0 for example in database]
    positive_examples = [example[-1] == 1 for example in database]

    weighted_neg = inner(negative_examples,weights)
    weighted_pos = inner(positive_examples,weights)

    total_weights = sum(weights)

    neg_ratio = float(weighted_neg / total_weights)
    pos_ratio = 1 - neg_ratio


#    neg_examples = float(sum(example[-1] == 0 for example in database))
    # note we are assuming binary classification

#    neg_ratio = neg_examples / len(database)
#    pos_ratio = 1 - neg_ratio

    return -1 * neg_ratio * log2(neg_ratio) - pos_ratio * log2(pos_ratio)


def information_gain(database, weights,attribute):
    total_entropy = entropy(database,weights)
    gain = total_entropy
    attr_index = database.ordered_attributes.index(attribute)

    for attr_level in database.attributes[attribute]:
        filtered_data = [ex for ex in database.data if ex[attr_index] == attr_level]
        gain -= entropy(filtered_data,weights) * len(filtered_data) / len(database)

    return gain


class DecisionStump(object):

    def __init__(database,weights):
        best_attribute = max(database.ordered_attributes[:-1], key=lambda x: information_gain(database,weights,x))

        self.database = database
        self.predictions = {}

        for attr_level in database.attributes[best_attribute]:
            filtered_data = [ex for ex in database.data if ex[attr_index] == attr_level]
            neg_examples = float(sum(example[-1] == 0 for example in database))
            self.predictions[attr_level] = (neg_examples < (len(database.data)/2))


    def predict(self,example):
        return self.database.attributes[-1][self.predictions[example[best_attribute]]]
