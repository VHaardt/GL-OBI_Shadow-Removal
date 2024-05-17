import matplotlib.pyplot as plt
import numpy as np
import json
import os
import argparse
import ipdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot metrics')
    parser.add_argument('--path', type=str, help='Path to the metrics file)')
    args = parser.parse_args()

    if args.path is None:
        experiments = os.listdir("checkpoints")
        # sort experiments by date of creation
        experiments.sort(key=lambda x: os.path.getctime(os.path.join("checkpoints", x)))
        args.path = os.path.join("checkpoints", experiments[-1], "metrics.json")

    with open(args.path, 'r') as file:
        metrics = json.load(file)


    metrics_names = []
    plots = []
    for metric, values in metrics.items():
        split = metric.split("_")[0]
        metric_name = metric.replace(split + "_", "")

        if metric_name not in metrics_names:
            metrics_names.append(metric_name)
            # create plot
            plt.figure()
            plt.title(metric_name)
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)

        # plot
        plt.plot(values, label=split)
        plt.legend()

    plt.show()



        



