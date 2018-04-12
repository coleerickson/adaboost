from math import exp
from math import log
from decision_stump import DecisionStump
from sys import argv
from parse_arff import Database
from argparse import ArgumentParser

def log2(x):
    return log(x,2)

sign = lambda x: 1 if x > 0 else -1
resign = lambda x: -1 if x == 0 else 1

def print_weights(weights):
    return(['{0:.4f}'.format(w) for w in weights])

def inner(x,y):
    return sum(xi*yi for (xi,yi) in zip(x,y))

class Adaboost:
    def __init__(self,weak_constructor,database,T):
        self.weak_constructor = weak_constructor
        self.database = database

        m = len(database.data)
        self.D = [1/m] * m
        self.alphas = [0] * T
        self.classifiers = [None] * T

        self._learn()

    def _learn(self):
#        all_preds = []
        for t in range(T):
#            print(print_weights(self.D))

            ht = self.weak_constructor(self.database,self.D)
            preds = [ht.predict(example) for example in self.database.data]
            wrong_preds = [ex[-1] != pred for ex,pred in zip(self.database.data, preds)]

            et = inner(self.D,wrong_preds) / sum(self.D)
#            print(et)
            if et == 0:
                alpha_t = 100
            else:
                alpha_t = log(((1-et)/et)) / 2

            signify = lambda x: 1 if x else -1
            self.D = [di * exp(alpha_t * signify(ex[-1] != pred)) for di,ex,pred in zip(self.D,self.database.data,preds)]
            z_t = sum(self.D) # normalization factor, so we sum to 1 (and are a distribution)
            self.D = [di / z_t for di in self.D]

            self.alphas[t] = alpha_t
            self.classifiers[t] = ht
#            all_preds.append(preds)

#            print('iter over')
#            print()
#            print()

#        each_pred_time_sequence = [[p[i] for p in all_preds] for i in range(len(self.database.data))]

#        for i,ex in enumerate(self.database.data):
#            print(each_pred_time_sequence[i], ex[-1])
#            print(int(sum(each_pred_time_sequence[i]) > (len(each_pred_time_sequence[i])/2)) == ex[-1])
#        print(self.alphas)



    def predict(self,example):
        return sign(sum(alpha_t * resign(ht.predict(example)) for alpha_t,ht in zip(self.alphas, self.classifiers)))
        # iz thiz right ???


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

    preds = [a.predict(ex) for ex in db.data]
    results = [sign(ex[-1]) == p for ex,p in zip(db.data, preds)]
    print(preds)
    print(sum(results) / len(results))
