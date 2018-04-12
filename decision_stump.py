from math import log
from collections import Counter

def log2(x):
    ''' Returns the base 2 logarithm of `x`. '''
    return log(x, 2)


def inner(x, y):
    ''' Returns he inner product of vectors `x` and `y`, where `x` and `y` are represented as lists. '''
    return sum(xi * yi for (xi, yi) in zip(x, y))


def entropy(examples, weights):
    '''
    Returns the entropy of `examples`. Entropy is defined in terms of the true class of the example. When counted, each of the `examples` is multiplied by the factor at the corresponding index in `weights`.
    '''
    negative_examples = [example[-1] == 0 for example in examples]
    positive_examples = [example[-1] == 1 for example in examples]

    if len(negative_examples) == 0 or len(positive_examples) == 0:
        return 0

    weighted_neg = inner(negative_examples, weights)
    weighted_pos = inner(positive_examples, weights)

    total_weights = sum(weights)

    neg_ratio = float(weighted_neg / total_weights)
    pos_ratio = 1 - neg_ratio

    return -neg_ratio * log2(neg_ratio) - pos_ratio * log2(pos_ratio)


def information_gain(database, weights, attribute):
    '''
    Computes the information gain of `database` by splitting on `attribute`. The examples in the database are reweighted by `weights`.
    '''
    total_entropy = entropy(database.data,weights)
    gain = total_entropy
    attr_index = database.ordered_attributes.index(attribute)

    # Computes split entropy
    for attr_level in range(len(database.attributes[attribute])):
        filtered_data = [ex for ex in database.data if ex[attr_index] == attr_level]
        gain -= entropy(filtered_data, weights) * len(filtered_data) / len(database)

    return gain


class DecisionStump:
    '''
    A classifier. A decision stump is a special case of a decision tree which has only one node. That is, a decision stump classifies based on only one attribute.
    '''

    def __init__(self, database, weights):
        ''' Learns/creates the decision stump by selecting the attribute that maximizes information gain. '''
        self.database = database

        # Select the attribute that maximizes information gain on the training set
        self.best_attribute = max(database.ordered_attributes[:-1], key=lambda x: information_gain(database, weights, x))

        # For each value of the best attribute, determine the majority class. In `self.predictions`, map that attribute value to the majority class.
        self.predictions = {}
        attr_index = database.ordered_attributes.index(self.best_attribute)
        for attr_level in range(len(database.attributes[self.best_attribute])):
            filtered_data = [ex for ex in database.data if ex[attr_index] == attr_level]
            neg_examples = float(sum(example[-1] == 0 for example in database.data))
            self.predictions[attr_level] = int(neg_examples < (len(database.data) / 2.0))


    def predict(self, example):
        ''' Returns the predicted class of `example` based on the attribute that maximized information gain at training time. '''
        attr_index = self.database.ordered_attributes.index(self.best_attribute)
        klass = self.database.ordered_attributes[-1]

        return self.database.attributes[klass][self.predictions[example[attr_index]]]
