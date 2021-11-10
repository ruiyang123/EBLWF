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
            dist[i, i] = 0.9
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
    domains = os.listdir(source)
    categories = os.listdir(os.path.join(source,domains[0]))
    for domain in domains:
        for cat in categories:
            imgs = os.listdir(os.path.join(source,domain,cat))
            for img in imgs:
                if np.random.rand() > 0.8:
                    row = {"path": os.path.join(source,domain,cat,img),"domain":domain,"class":cat, "type":"test"}
                else:
                    row = {"path": os.path.join(source, domain, cat, img), "domain": domain, "class": cat,
                           "type": "train"}
                df = df.append(row,ignore_index=True)

    df.to_csv("./PACS_info.csv",index=False)



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
    #source = "./datasets/PACS"
    #get_dataset_info(source)

    # DIL
    Pacs_task_order = {"photo": 1, "cartoon": 2, "art_painting": 3, "sketch": 4}
    info_path = "./PACS_info.csv"
    pacs_info = pd.read_csv(info_path)
    df = simple_domain_split(pacs_info, task_order=Pacs_task_order)
    df.to_csv("./PACS_DIL.csv")

    # smooth-DIL

    df =  domain_split(df=pacs_info,task_num=6,mode="random")
    df.to_csv("./PACS_smoothDIL1.csv")

    df = domain_split(df=pacs_info, task_num=6, mode="random")
    df.to_csv("./PACS_smoothDIL2.csv")

    df = domain_split(df=pacs_info, task_num=6, mode="random")
    df.to_csv("./PACS_smoothDIL3.csv")
    
    # CIL
    df = class_split(df=pacs_info,task_num=7,mode="class_inc")
    df.to_csv("./PACS_CIL.csv")

    # smooth-CIL
    df = class_split(df=pacs_info,task_num=6,mode="random")
    df.to_csv("./PACS_smoothCIL1.csv")

    df = class_split(df=pacs_info,task_num=6,mode="random")
    df.to_csv("./PACS_smoothCIL2.csv")

    df = class_split(df=pacs_info,task_num=6,mode="random")
    df.to_csv("./PACS_smoothCIL3.csv")









