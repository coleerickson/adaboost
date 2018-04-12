from math import log
from collections import Counter

def log2(x):
    return log(x,2)

def inner(x,y):
    return sum(xi*yi for (xi,yi) in zip(x,y))

def entropy(examples,weights):
    negative_examples = [example[-1] == 0 for example in examples]
    positive_examples = [example[-1] == 1 for example in examples]

    if len(negative_examples) == 0 or len(positive_examples) == 0:
        return 0

    weighted_neg = inner(negative_examples,weights)
    weighted_pos = inner(positive_examples,weights)

    total_weights = sum(weights)

    neg_ratio = float(weighted_neg / total_weights)
    pos_ratio = 1 - neg_ratio

    return -1 * neg_ratio * log2(neg_ratio) - pos_ratio * log2(pos_ratio)


def information_gain(database, weights,attribute):
    total_entropy = entropy(database.data,weights)
    gain = total_entropy
    attr_index = database.ordered_attributes.index(attribute)

    for attr_level in range(len(database.attributes[attribute])):
        filtered_data = [ex for ex in database.data if ex[attr_index] == attr_level]
        gain -= entropy(filtered_data,weights) * len(filtered_data) / len(database)

    return gain


class DecisionStump:

    def __init__(self,database,weights):
        self.best_attribute = max(database.ordered_attributes[:-1], key=lambda x: information_gain(database,weights,x))

        self.database = database
        self.predictions = {}

        attr_index = database.ordered_attributes.index(self.best_attribute)

        for attr_level in range(len(database.attributes[self.best_attribute])):
            filtered_data = [ex for ex in database.data if ex[attr_index] == attr_level]
            neg_examples = float(sum(example[-1] == 0 for example in database.data))
            self.predictions[attr_level] = int(neg_examples < (len(database.data)/2))


    def predict(self,example):
        attr_index = self.database.ordered_attributes.index(self.best_attribute)
        klass = self.database.ordered_attributes[-1]

        return self.database.attributes[klass][self.predictions[example[attr_index]]]
