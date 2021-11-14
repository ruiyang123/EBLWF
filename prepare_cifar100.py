import os
import numpy as np
import pandas as pd
import shutil
import glob

def get_distribution(task_num,categories,mode):
    distribution = {}
    if mode=="uniform":
        for cat in categories:
            distribution[cat] = [1./task_num]*task_num
    elif mode == "class_inc":
        dist = np.ones((len(categories),task_num))*(0.1/(task_num-1))
        for i,cat in enumerate(categories):
            dist[i, int(i/2)] = 0.9
            distribution[cat] = list(dist[i,:])
    elif mode == "random":
        for cat in categories:
            dist = [np.random.random() for i in range(task_num)]
            k = sum(dist)
            dist = [i/k for i in dist]
            distribution[cat] = dist
    return distribution




def get_dataset_info(source):
    df = pd.DataFrame()
    train_path = os.path.join(source,"train")
    test_path = os.path.join(source,"test")

    class_names = os.listdir(train_path)

    for cla in class_names:
        domains = os.listdir(os.path.join(train_path,cla))
        for i,d in enumerate(domains):
            imgs = os.listdir(os.path.join(train_path,cla,d))
            for img in imgs:
                row = {"path":os.path.join(train_path,cla,d,img),"domain":"domain{}".format(i),"class":cla,"type":"train"}
                df = df.append(row,ignore_index=True)

    for cla in class_names:
        domains = os.listdir(os.path.join(test_path,cla))
        for i,d in enumerate(domains):
            imgs = os.listdir(os.path.join(test_path,cla,d))
            for img in imgs:
                row = {"path":os.path.join(test_path,cla,d,img),"domain":"domain{}".format(i),"class":cla,"type":"test"}
                df = df.append(row,ignore_index=True)

    df.to_csv("./cifar100_info.csv",index=False)



def simple_domain_split(df,task_order=None):
    if task_order:
        df["task"] = df["domain"].map(task_order)
    else:
        domains = list(df["domain"].unique())
        def apply_domain(x):
            for i, domain in enumerate(domains):
                if x==domain:
                    return i+1
        df["task"] = df["domain"].apply(apply_domain)
    return df



def class_split(df,task_num,mode):
    classes = list(df["class"].unique())
    dist_train = get_distribution(task_num,classes,mode)
    dist_test = get_distribution(task_num,classes,mode="uniform")

    dists = {}
    dists["train"] = dist_train
    dists["test"] = dist_test

    def class_dist_split(series, dists):
        class_ = series["class"]
        type_ = series["type"]
        dist_ = dists[type_][class_]
        a = np.random.random()

        start = 0
        stop = 0
        for i, d in enumerate(dist_):
            stop = start + d
            if a > start and a < stop:
                return i + 1
            start = stop
    df["task"] = df.apply(class_dist_split,args = (dists,),axis=1)
    return df


def domain_split(df,task_num,mode):
    domains= list(df["domain"].unique())
    dist_train = get_distribution(task_num,domains,mode)
    dist_test = get_distribution(task_num,domains,mode="uniform")

    dists = {}
    dists["train"] = dist_train
    dists["test"] = dist_test

    def domain_dist_split(series, dists):
        domain_ = series["domain"]
        type_ = series["type"]
        dist_ = dists[type_][domain_]
        a = np.random.random()

        start = 0
        stop = 0
        for i, d in enumerate(dist_):
            stop = start + d
            if a > start and a < stop:
                return i + 1
            start = stop
    df["task"] = df.apply(domain_dist_split,args = (dists,),axis=1)
    return df


def get_multi_var_distribution(task_num,classes,domains_,mode):
    if mode == "random":
        distribution = {}
        dist = np.random.random((task_num,len(classes),len(domains_)))
        dist /= np.sum(dist,axis=0)
        for i,cl in enumerate(classes):
            distribution[cl] = {}
            for j,do in enumerate(domains_):
                distribution[cl][do] = dist[:,i,j]
        return distribution
    elif mode == "uniform":
        distribution = {}
        dist = np.ones((task_num,len(classes),len(domains_)))*1./task_num
        for i,cl in enumerate(classes):
            distribution[cl] = {}
            for j,do in enumerate(domains_):
                distribution[cl][do] = dist[:,i,j]
        return distribution


def create_class_domain_split(df,task_num,mode):
    classes = list(df["class"].unique())
    domains = list(df["domain"].unique())
    dist_train = get_multi_var_distribution(task_num,classes,domains,mode)
    dist_test = get_multi_var_distribution(task_num,classes,domains,mode="uniform")

    def dist_domain_class_task(series, dists):
        class_ = series["class"]
        type_ = series["type"]
        domain_ = series["domain"]
        dist_ = dists[type_][class_][domain_]
        a = np.random.random()

        start = 0
        stop = 0
        for i, d in enumerate(dist_):
            stop = start + d
            if a > start and a < stop:
                return i + 1
            start = stop

    dist_all = {}
    dist_all["train"] = dist_train
    dist_all["test"] = dist_test
    df["task"] = df.apply(dist_domain_class_task, args=(dist_all,), axis=1)
    return df


if __name__ == "__main__":
    source = "./datasets/cifar100"
    get_dataset_info(source)

    # DIL
    cifar100_task_order = {"domain0": 1, "domain1": 2, "domain2": 3, "domain3": 4, "domain4": 5}
    info_path = "./cifar100_info.csv"
    cifar_info = pd.read_csv(info_path)
    df = simple_domain_split(cifar_info,task_order=cifar100_task_order)
    df.to_csv("./cifar_DIL.csv")

    # smooth-DIL

    df = domain_split(df=cifar_info,task_num=8,mode="random")
    df.to_csv("./cifar_smoothDIL1.csv")

    df = domain_split(df=cifar_info,task_num=8,mode="random")
    df.to_csv("./cifar_smoothDIL2.csv")

    df = domain_split(df=cifar_info,task_num=8,mode="random")
    df.to_csv("./cifar_smoothDIL3.csv")

    # CIL
    df =  class_split(df=cifar_info,task_num=10,mode="class_inc")
    df.to_csv("./cifar_class_imb.csv")

    # smooth-CIL
    df = class_split(df=cifar_info,task_num=8,mode="random")
    df.to_csv("./cifar_smoothCIL1.csv")

    df = class_split(df=cifar_info,task_num=8,mode="random")
    df.to_csv("./cifar_smoothCIL2.csv")

    df = class_split(df=cifar_info,task_num=8,mode="random")
    df.to_csv("./cifar_smoothCIL3.csv")







