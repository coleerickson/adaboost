from math import exp
from math import log
from decision_stump import DecisionStump
from sys import argv
from parse_arff import Database
from argparse import ArgumentParser


def log2(x):
    ''' Returns the base 2 logarithm of `x`. '''
    return log(x, 2)


def sign(x): return 1 if x > 0 else -1


def resign(x): return -1 if x == 0 else 1


def print_weights(weights):
    return(['{0:.4f}'.format(w) for w in weights])


def inner(x, y):
    ''' Returns the inner product (dot product) of vectors `x` and `y`, where `x` and `y` are represented as lists. '''
    return sum(xi * yi for (xi, yi) in zip(x, y))


class Adaboost:
    def __init__(self, weak_constructor, database, T):
        '''
        Constructs and learns a classifier by "boosting" another classifier as described by Freund and Schapire.

        `weak_constructor` -- the constructor for the classifier learning algorithm which will be boosted.
            This object must have:
            * a constructor which takes as arguments a `Database` and a list of weights for each of the examples
              in the database
            * a predict method which takes as its argument an example, represented by a list of attribute values,
              and returns the predicted class for the example
        `database` -- the training set
        `T` -- the number of boosting rounds
        '''

        self.weak_constructor = weak_constructor
        self.database = database

        m = len(database.data)
        self.D = [1 / m] * m
        self.alphas = [0] * T
        self.classifiers = [None] * T

        self._learn()

    def _learn(self):
        '''
        Learns the classifier. Note that the notation is borrowed from Freund and Schapire's pseudocode on p. 3
        of "A Short Introduction to Boosting."
        '''
        for t in range(T):
            ht = self.weak_constructor(self.database, self.D)
            preds = [ht.predict(example) for example in self.database.data]
            wrong_preds = [ex[-1] != pred for ex, pred in zip(self.database.data, preds)]

            et = inner(self.D, wrong_preds) / sum(self.D)

            if et == 0:
                alpha_t = 10
            else:
                alpha_t = log(((1 - et) / et)) / 2

            def signify(x): return 1 if x else -1
            self.D = [di * exp(alpha_t * signify(ex[-1] != pred))
                      for di, ex, pred in zip(self.D, self.database.data, preds)]
            z_t = sum(self.D)  # normalization factor, so we sum to 1 (and are a distribution)
            self.D = [di / z_t for di in self.D]

            self.alphas[t] = alpha_t
            self.classifiers[t] = ht

    def predict(self, example):
        ''' Predicts the class of the example. '''
        return sign(sum(alpha_t * resign(ht.predict(example)) for alpha_t, ht in zip(self.alphas, self.classifiers)))


if __name__ == '__main__':
    parser = ArgumentParser(description='Adaboost')

    parser = ArgumentParser()
    parser.add_argument('-f', '--file', help='.arff file name', required=True)
    parser.add_argument('-i', '--iters', type=int, help='number of classifiers to train', required=True)

    args = parser.parse_args()

    file_name = args.file
    T = args.iters

    db = Database()
    db.read_data(file_name)

    a = Adaboost(DecisionStump, db, T)

    preds = [a.predict(ex) for ex in db.data]
    results = [sign(ex[-1]) == p for ex, p in zip(db.data, preds)]
    print(sum(results) / len(results))
