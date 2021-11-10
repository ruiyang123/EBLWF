import numpy as np
import pandas as pd
import os
import argparse


parser = argparse.ArgumentParser(description="Post-Process")
parser.add_argument('--dataset', type="str")








def backward_f(result_table):
    bf = 0.
    for task,data in enumerate(result_table):
        bf += data[task] - result_table[-1,task]

    return -bf/(len(result_table)-1)




if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset
    scenarios = {
        "DIL": ["{}_DIL_result".format(dataset)],
        "smooth-DIL": ["{}_smoothDIL1_result".format(dataset), "{}_smoothDIL2_result".format(dataset), "{}_smoothDIL3_result".format(dataset)],
        "CIL": ["{}_CIL_result".format(dataset)],
        "smooth-CIL": ["{}_smoothCIL1_result".format(dataset), "{}_smoothCIL2_result".format(dataset),
                       "{}_smoothCIL3_result".format(dataset)]
    }

    methods = ["autolwf", "cul", "er", "lwf", "mlwf", "naive", "oewc", "si"]


    results = {}
    for sce in scenarios.keys():
        results[sce] = {}
        for method in methods:
            results[sce][method] = {}
            results[sce][method]["acc"] = []
            results[sce][method]["bf"] = []

    for sce, result_folders in scenarios.items():
        for result_folder in result_folders:
            base_folder = os.path.join("./", result_folder)
            for method in methods:
                method_folder = os.path.join(base_folder, method)
                experiments = os.listdir(method_folder)
                for experiment in experiments:
                    result_files = os.listdir(os.path.join(method_folder, experiment))
                    result_csv = [file for file in result_files if ".csv" in file]
                    result = pd.read_csv(os.path.join(method_folder, experiment, result_csv[0]), header=None)
                    result = result.values
                    task_num = len(result)
                    av_accuracy = np.mean(result, axis=1)
                    results[sce][method]['acc'].append(av_accuracy)
                    bf = backward_f(result)
                    results[sce][method]['bf'].append(bf)

    for sce in scenarios.keys():
        for method in methods:
            bf = results[sce][method]['bf']
            av_bf = np.mean(bf)
            std_bf = np.std(bf)

            acc = np.array(results[sce][method]['acc'])
            av_acc = np.mean(acc,0)[-1]
            std_acc = np.std(acc,0)[-1]

            print(sce,method,"average BF: {}".format(av_bf), "std-BF: {}".format(std_bf), "average Acc: {}".format(av_acc),"std-Acc: {}".format(std_acc))