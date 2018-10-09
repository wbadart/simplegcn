#!/usr/bin/env python3

"""
simplegcn/__main__.py

Main entry point for running the simplegcn module as a command. Reads a
serialized graph from disk, trains a GCN based on it, and evaluates that model.

Will Badart <badart_william (at) bah (dot) com
created: OCT 2018
"""

def main():
    """
    Main function to run from command line. Install the simplegcn package with
    pip and then run `simplegcn --help` to see command line usage.
    """
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Create, train, and evaluate a GCN')
    parser.add_argument('-e', '--epochs', type=int, default=500,
                        help='Number of training epochs to run (default:500)')
    args = parser.parse_args()
    print(args.epochs)


if __name__ == '__main__':
    main()
