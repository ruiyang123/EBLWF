import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

dataset = "cifar"


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
        results[sce][method] = []


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
                results[sce][method].append(av_accuracy)
                print(sce,result_folder,method,experiment,av_accuracy)

np.save("./results_cifar.npy",results)


results = np.load('results_cifar.npy',allow_pickle='TRUE').item()
df = pd.DataFrame()


for sce in scenarios.keys():
    for method in methods:
        all_ex_res = np.array(results[sce][method])
        av_ = np.mean(all_ex_res,0)
        std_ = np.std(all_ex_res,0)
        results[sce][method] = av_

        for task in range(len(av_)):
            if method=="erlwf":
                method="ER-lwf"
            row = {"method":method,"task_num":int(task),"Mean-accuracy":av_[task],"scenario":sce}
            df = df.append(row,ignore_index=True)



paper_rc = {'lines.linewidth': 4, 'lines.markersize': 10}
sns.set_context("paper", rc = paper_rc)
fig, axs = plt.subplots(1,4,figsize=(18,6))

sns.lineplot(ax=axs[0] ,x="task_num",y="Mean-accuracy",hue="method",
                        style="method",markers=True,data=df[df["scenario"]=="DIL"])
axs[0].get_legend().remove()
axs[0].set_title("DIL",fontsize=20)
axs[0].set(xlabel=None)
axs[0].set_ylabel("Average-Accuracy",fontsize=20)
axs[0].xaxis.set_ticks(np.arange(0,5,1))
sns.lineplot(ax=axs[1],x="task_num",y="Mean-accuracy",hue="method",
                        style="method",markers=True,data=df[df["scenario"]=="smooth-DIL"])

axs[1].get_legend().remove()
axs[1].set_title("smooth-DIL",fontsize=20)
axs[1].set(xlabel=None)
axs[1].set(ylabel=None)
axs[1].set(xlabel=None)
axs[1].xaxis.set_ticks(np.arange(0,8,1))

sns.lineplot(ax=axs[2],x="task_num",y="Mean-accuracy",hue="method",
                        style="method",markers=True,data=df[df["scenario"]=="CIL"])
axs[2].get_legend().remove()
axs[2].set_title("CIL",fontsize=20)
axs[2].set(xlabel=None)
axs[2].set(ylabel=None)
axs[2].set(xlabel=None)
axs[2].xaxis.set_ticks(np.arange(0,10,1))


sns.lineplot(ax=axs[3],x="task_num",y="Mean-accuracy",hue="method",
                        style="method",markers=True,data=df[df["scenario"]=="smooth-CIL"])
axs[3].set_title("smooth-CIL",fontsize=20)
axs[3].set(xlabel=None)
axs[3].set(ylabel=None)
axs[3].set(xlabel=None)
axs[3].xaxis.set_ticks(np.arange(0,8,1))
axs[3].get_legend().remove()

handles, labels = axs[3].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',ncol=8,fontsize=16)

for ax in axs.flat:
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
fig.text(0.2, 0.1, 'Task', ha='center',fontsize=18)
plt.subplots_adjust(bottom=0.2,wspace=0.25,hspace=0)

fig.savefig("./cifar_result.png")


