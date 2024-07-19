import matplotlib.pyplot as plt
import numpy as np
import json
import os
import argparse
import ipdb

def insert_nulls_between_entries(original_list, len2):
    len1 = len(original_list)
    if len1 >= len2:
        return original_list
    
    # Calcola il numero di valori nulli da inserire tra ogni entrata
    num_nulls = (len2 - len1) // (len1 - 1)
    
    new_list = []
    
    for i in range(len1 - 1):
        new_list.append(original_list[i])
        new_list.extend([None] * num_nulls)
    
    # Aggiungi l'ultimo elemento della lista originale
    new_list.append(original_list[-1])
    
    # Se c'è un resto, aggiungilo alla fine
    remainder = (len2 - len1) % (len1 - 1)
    if remainder > 0:
        new_list.extend([None] * remainder)
    
    return new_list


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
    c = 0
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
        
        if split == 'train':
            c = len(values)
        # plot
        if c != 0:
            values = insert_nulls_between_entries(values, c)
            x = [i for i, point in enumerate(values) if point is not None]
            plt.plot(x, [point for point in values if point is not None], label=split)

        else:
            plt.plot(values, label=split)
        
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.show()



        



