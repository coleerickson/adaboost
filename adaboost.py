from math import exp
from math import log
from decision_stump import DecisionStump
from sys import argv
from parse_arff import Database
from argparse import ArgumentParser

def log2(x):
    return log(x,2)

sign = lambda x: 1 if x > 0 else -1


def inner(x,y):
    return sum(xi*yi for (xi,yi) in zip(x,y))

class Adaboost:
    def __init__(self,weak_constructor,database,T):
        self.weak_constructor = weak_constructor
        self.database = database


        m = len(database.data)
#        D = [ [1/m] * m ]
        D = [1/m] * m
        self.alphas = [0] * T
        self.classifiers = [None] * T

        for t in range(T):
            ht = self.weak_constructor(self.database,D)
            preds = [ht.predict(example) for example in self.database.data]

            wrong_preds = [ex[-1] == pred for ex,pred in zip(self.database.data, preds)]
            et = inner(D,wrong_preds) / sum(D)

            if et == 0:
                alpha_t = 100
            else:
                alpha_t = log((1-et)/et) / 2

            signify = lambda x: 1 if x else -1

            D = [di * exp(alpha_t * signify(ex[-1] == pred)) for di,ex,pred in zip(D,self.database.data,preds)]

            z_t = sum(D) # normalization factor, so we sum to 1
            D = [di / z_t for di in D]

            self.alphas[t] = alpha_t
            self.classifiers[t] = ht



    def predict(self,example):
        return sign(sum(alpha_t * ht.predict(example) for alpha_t,ht in zip(self.alphas, self.classifiers)))


if __name__ == '__main__':
    parser = ArgumentParser(description='Adaboost')

    parser = ArgumentParser()
    parser.add_argument('-f', '--file', help='.arff file name')
    parser.add_argument('-i', '--iters', type=int, help='number of classifiers to train')

    args = parser.parse_args()

    file_name = args.file
    T = args.iters

    db = Database()
    db.read_data(file_name)


    a = Adaboost(DecisionStump,db,T)
