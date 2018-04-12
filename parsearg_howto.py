from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-f', '--file')
parser.add_argument('-i', '--iters', type=int)

args = parser.parse_args()
print(args)
