# AdaBoost and Decision Stumps

### Usage

To run, pass an ARFF dataset and a number of boosting rounds. For example:

```sh
$ python3 adaboost.py --file weather.nominal.arff --iters 50
```

Note that, when run directly, `adaboost.py` is hardcoded to use decision stumps as the underlying "weak" classifier learning algorithm.

### Project Layout

`adaboost.py` implements the AdaBoost algorithm from Freund and Schapire.

`decision_stump.py` implements decision stumps, a simple classifier learning algorithm based on decision trees.

`parse_arff.py` is used for parsing ARFF files.
