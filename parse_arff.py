# A library for reading .arff data files
import re
import random
from typing import *
from copy import copy
from math import ceil


class Database:

    def __init__(self):
        self.data = []
        self.attributes = {}
        self.ordered_attributes = []

    def read_data(self,file_name:str):
        '''Parses the passed ARFF file into a usable object'''

        # Read lines from file
        with open(file_name) as f:
            lines = f.readlines()

        # Filter and trim lines so that they're ready for further parsing
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line != '' and line[0] != '%']

        self.parse_relation(lines)
        self.parse_attributes(lines)
        self.parse_data(lines)

    def hold_one_out(self):
        '''
        A generator of tuples for use in validation. The first element of each
        tuple is the training set -- all the examples but one. The second is
        the single example for use as the test set.
        '''
        return self.k_fold(len(self.data))

    def k_fold(self, k):
        '''
        A generator of tuples for use in k-fold cross-validation. The first
        element of each tuple is the training set; the second, the test set.
        The total number of tuples is k. No example appears in more than one
        testing set.
        '''
        for choice in range(k):
            # Take index of each data entry mod k
            # for each i in 0..k-1 denoting test choice
            # choose (yield) the combo train,test

            train = Database()
            train.attributes = self.attributes
            train.ordered_attributes = self.ordered_attributes

            test = Database()
            test.attributes = self.attributes
            test.ordered_attributes = self.ordered_attributes

            train.data = [d for i, d in enumerate(
                self.data) if i % k != choice]
            test.data = [d for i, d in enumerate(self.data) if i % k == choice]
            yield train, test

    def train_test_split(self, p=0.8, seed=None):
        '''
        Split the data into train and test data sets.
        @param p: the proportion of the data you want to use for training
        '''
        if p == 1:
            raise Exception(
                'Don\'t use this to train on all your data, silly! You already have it!')

        random.seed(seed)

        # Randomly pick the test indices -- the (1 - p) fraction of them
        test_indices = set(random.sample(
            range(len(self.data)), int(len(self.data) * (1 - p))))

        # Generate the training set and test set using those indices
        train_set = [d for i, d in enumerate(
            self.data) if i not in test_indices]
        test_set = [d for i, d in enumerate(self.data) if i in test_indices]


        train = Database()
        train.attributes = self.attributes
        train.ordered_attributes = self.ordered_attributes

        test = Database()
        test.attributes = self.attributes
        test.ordered_attributes = self.ordered_attributes

        return train_db, test_db

    def parse_relation(self, lines: List):
        '''Parse the relation section of the ARFF file'''
        first_line = lines[0].split(' ', 1)
        if len(first_line) != 2:
            raise Exception('Data formatted incorrectly, length wrong')

        if first_line[0].lower() != '@relation':
            raise Exception('Expected @relation, got {}'.format(first_line[0]))

        self.name = first_line[1]
        lines.pop(0)

    def parse_attributes(self, lines: List):
        '''Parse the attributes section of the ARFF file'''
        self.attributes = {}
        self.ordered_attributes = []

        if len(lines) == 0:
            raise Exception("No attributes found in file")

        # Parse each attribute line
        line = lines[0]
        while line.lower().startswith('@attribute'):
            # Remove the line from the list so that we don't process it again
            lines.pop(0)

            # Use a regular expression to match the attribute name and values
            match = re.match(
                '\s*([^\s]*)\s*\{(.*)\}$', line[len('@attribute'):])
            assert bool(
                match), 'Expected regex match on attribute line "%s"' % line

            attr_name = match.group(1)
            attr_values = [x.strip() for x in match.group(2).split(',')]

            self.attributes[attr_name] = attr_values
            self.ordered_attributes.append(attr_name)

            # Go on to the next line
            line = lines[0]

        # The last attribute we parse is considered the output variable
        self.output_var = attr_name

    def parse_data(self, lines: List):
        '''Parse the data section of the ARFF file'''
        first_line = lines[0]
        if first_line.lower() != '@data':
            raise Exception('Expected @data, got {}'.format(first_line))
        lines.pop(0)

        split_lines = [l.split(',') for l in lines]
        split_clean_lines = [[s.strip() for s in l] for l in split_lines]
#        self.data = [dict(zip(self.ordered_attributes, sl))
#                     for sl in split_clean_lines]

        self.data = [[self.get_attr_index(i,v) for i,v in enumerate(sl)]
                    for sl in split_clean_lines]
#        self.validate_data()

    def get_attr_index(self,index,val):
        if val == '?':
            return None
        else:
            return self.attributes[self.ordered_attributes[index]].index(val)
            # May throw error if data invalid.

    def __len__(self):
        return len(self.data)
#    def validate_data(self):
#        '''Check that all examples have valid values'''
#        for entry in self.data:
#            for k, v in entry.items():
#                assert v == '?' or v in self.attributes[k]

    def __str__(self):
        attr_rep = list(self.ordered_attributes)

        # The first 5 examples in the database
        data_rep = '\n\t'.join([str(s)
                                for s in self.data[:min(5, len(self.data))]])

        return '''name:
\t{}

attributes:
\t{}

example data:
\t{}'''.format(self.name, attr_rep, data_rep)


if __name__ == "__main__":
    # Sanity tests
    d = Database()
    d.read_data('./NominalData/weather.nominal.arff')
    assert d.name == 'weather.symbolic'
    print(d)


    s = Database()
    s.read_data('./NominalData/soybean.arff')
    assert s.name == 'soybean'
    print(s)

    c = Database()
    c.read_data('./NominalData/contact-lenses.arff')
    assert c.name == 'contact-lenses'
    print(c)

    t = Database()
    t.read_data('./NominalData/titanic.arff')
    assert t.name == 'titanic.survival'
    print(t)
