import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(results_path):
    archs = defaultdict(dict)

    for filename in os.listdir(results_path):
        arch, hyperparams = filename.split(".", 1)
        hyperparams = hyperparams.replace(".tsv", "")
        archs[arch][hyperparams] = os.path.join(results_path, filename)

    for arch in archs.keys():
        plot_arch(archs[arch])

def plot_arch(models):

    model_dfs = dict()
    min_length = 50
    for model, path in models.items():
        df = pd.read_csv(path, sep="\t")
        if len(df) < min_length: min_length = len(df)
        model_dfs[model] = df

    for model, df in model_dfs.items():
        print model
        df = df[:min_length]
        plt.plot(df["train perpl"])

    plt.savefig("example.pdf")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot model results.')
    parser.add_argument('--results', metavar='P', required=True,
                        help='Location of results directory.')
    args = parser.parse_args()
    plot(args.results)
    

if __name__ == "__main__":
    main()
