class Adaboost:
    def __init__(self,weak_constructor,database):
        self.weak_constructor = weak_constructor
        self.database = database


        m = len(database.data)
        D = [ [1/m] * m ]
        T = 1000 # Here be dragons


        for _ in range(T):
            # Make new databas????
            D[-1]
            iter_classifier = self.weak_constructor(self.database)


    def predict(self,example):
