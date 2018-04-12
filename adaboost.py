from math import exp
from math import log

def log2(x):
    return log(x,2)

sign = lambda x: 1 if x > 0 else -1


def inner(x,y):
    return sum(xi*yi for (xi,yi) in zip(x,y))

class Adaboost:
    def __init__(self,weak_constructor,database):
        self.weak_constructor = weak_constructor
        self.database = database


        m = len(database.data)
#        D = [ [1/m] * m ]
        D = [1/m] * m
        T = 1000 # Here be dragons
        self.alphas = [0] * T
        self.classifiers = [None] * T

        for t in range(T):
            # Make new databas????
#            D[-1]
            ht = self.weak_constructor(self.database)
            preds = [ht.predict(example) for example in self.database.data]

            wrong_preds = [ex[-1] == pred for ex,pred in zip(self.database.data, preds)]
            et = inner(D,wrong_preds) / sum(D)
            alpha_t = log((1-et)/et) / 2



            signify = lambda x: 1 if x else -1

#            for di, example, pred in zip(D,examples,preds):
#                di / zt * exp(alpha_t * signify(example[-1] == pred))

#            D = [d_i * exp(-1 * alpha_t * signify(ex[-1]) * signify(pred)) / zt for di,ex,pred in zip(D,self.database.data,preds)]
            D = [di * exp(alpha_t * signify(ex[-1] == pred)) for di,ex,pred in zip(D,self.database.data,preds)]

            z_t = sum(D) # normalization factor, so we sum to 1
            D = [di / z_t for di in D]

            self.alphas[t] = alpha_t
            self.classifiers[t] = ht



    def predict(self,example):
        return sign(sum(alpha_t * ht.predict(example) for alpha_t,ht in zip(self.alphas, self.classifiers)))
