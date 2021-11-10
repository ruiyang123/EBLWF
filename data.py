import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
import torch
import glob
from PIL import Image
import pickle
import os

my_transform = transforms.Compose([transforms.ToTensor()])
datapath = ['bottle_01', 'bottle_02', 'bottle_03', 'bottle_04', 'bowl_01', 'bowl_02', 'bowl_03', 'bowl_04', 'bowl_05',
            'corkscrew_01', 'cottonswab_01', 'cottonswab_02', 'cup_01', 'cup_02', 'cup_03', 'cup_04', 'cup_05',
            'cup_06', 'cup_07', 'cup_08', 'cup_10', 'cushion_01', 'cushion_02', 'cushion_03', 'glasses_01',
            'glasses_02', 'glasses_03', 'glasses_04', 'knife_01', 'ladle_01', 'ladle_02', 'ladle_03', 'ladle_04',
            'mask_01', 'mask_02', 'mask_03', 'mask_04', 'mask_05', 'paper_cutter_01', 'paper_cutter_02',
            'paper_cutter_03', 'paper_cutter_04', 'pencil_01', 'pencil_02', 'pencil_03', 'pencil_04', 'pencil_05',
            'plasticbag_01', 'plasticbag_02', 'plasticbag_03', 'plug_01', 'plug_02', 'plug_03', 'plug_04', 'pot_01',
            'scissors_01', 'scissors_02', 'scissors_03', 'stapler_01', 'stapler_02', 'stapler_03', 'thermometer_01',
            'thermometer_02', 'thermometer_03', 'toy_01', 'toy_02', 'toy_03', 'toy_04', 'toy_05','nail_clippers_01','nail_clippers_02',
            'nail_clippers_03', 'bracelet_01', 'bracelet_02','bracelet_03', 'comb_01','comb_02',
            'comb_03', 'umbrella_01','umbrella_02','umbrella_03','socks_01','socks_02','socks_03',
            'toothpaste_01','toothpaste_02','toothpaste_03','wallet_01','wallet_02','wallet_03',
            'headphone_01','headphone_02','headphone_03', 'key_01','key_02','key_03',
             'battery_01', 'battery_02', 'mouse_01', 'pencilcase_01', 'pencilcase_02', 'tape_01',
             'chopsticks_01', 'chopsticks_02', 'chopsticks_03',
               'notebook_01', 'notebook_02', 'notebook_03',
               'spoon_01', 'spoon_02', 'spoon_03',
               'tissue_01', 'tissue_02', 'tissue_03',
              'clamp_01', 'clamp_02', 'hat_01', 'hat_02', 'u_disk_01', 'u_disk_02', 'swimming_glasses_01'
            ]

taco_path = ["Can", "Bottle","Plastic bag & wrapper", "Cigarette","Carton","Cup"]

trash_path = ["carton","metal","paper","plastic","verre"]

trash_dist_path = ["carton","metal","papier","plastique","verre"]

map_labels_PACS = {"dog":0,"elephant":1,"giraffe":2,"guitar":3,"horse":4,"house":5,"person":6}

map_labels_cifar100 = {'aquatic_mammals': 0,
'fish': 1,
'flowers': 2,
'food_containers': 3,
'fruit_and_vegetables': 4,
'household_electrical_devices': 5,
'household_furniture': 6,
'insects': 7,
'large_carnivores': 8,
'large_man-made_outdoor_things': 9,
'large_natural_outdoor_scenes': 10,
'large_omnivores_and_herbivores': 11,
'medium_mammals': 12,
'non-insect_invertebrates': 13,
'people': 14,
'reptiles': 15,
'small_mammals': 16,
'trees': 17,
'vehicles_1': 18,
'vehicles_2': 19}

class TacoDataset(Dataset):

    def __init__(self, batch_num, mode='train', own_transform=None, factor='taco_dataset',base_path=r"/data/yangr/taco/"):
        batch_num += 1
        self.transform = own_transform
        if mode == 'train':
            self.imgs = []
            self.labels = []
            for i in range(len(taco_path)):
                temp = glob.glob(base_path + factor + '/train/task{}/{}/*.png'.format(batch_num, taco_path[i]))
                print(taco_path[i])

                self.imgs.extend([Image.open(x).convert('RGB').resize((224, 224)) for x in temp])
                self.labels.extend([i] * len(temp))
            print("  --> batch{}'-dataset consisting of {} samples".format(batch_num, len(self)))
        else:
            self.imgs = []
            self.labels = []
            for i in range(len(taco_path)):
                print(taco_path[i])
                temp = glob.glob(base_path + factor + '/test/task{}/{}/*.png'.format(batch_num,taco_path[i]))
                self.imgs.extend([Image.open(x).convert('RGB').resize((224, 224)) for x in temp])
                self.labels.extend([i] * len(temp))
            print("  --> test'-dataset consisting of {} samples".format(len(self)))

    def __setitem__(self, index, value):
        self.imgs[index] = value[0]
        self.labels[index] = value[1]

    def __getitem__(self, index):

        fn = self.imgs[index]
        label = self.labels[index]
        img = fn

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):

        return len(self.imgs)



class PACSDataset(Dataset):

    def __init__(self, batch_num,df, mode='train', own_transform=None):
        batch_num += 1
        self.transform = own_transform
        df["label"] = df["class"].map(map_labels_PACS)

        if mode == 'train':
            self.imgs = []
            self.labels = []
            temp = list(df["path"][(df["task"] == batch_num)&(df["type"] == "train")])
            labels = list(df["label"][(df["task"] == batch_num)&(df["type"] == "train")])


            self.imgs.extend([Image.open(x).convert('RGB').resize((64, 64)) for x in temp])
            self.labels.extend(labels)
            print("  --> batch{}'-dataset consisting of {} samples".format(batch_num, len(self)))
        else:
            self.imgs = []
            self.labels = []
            temp = list(df["path"][(df["task"] == batch_num) & (df["type"] == "test")])
            labels = list(df["label"][(df["task"] == batch_num) & (df["type"] == "test")])

            self.imgs.extend([Image.open(x).convert('RGB').resize((64, 64)) for x in temp])
            self.labels.extend(labels)
            print("  --> test'-dataset consisting of {} samples".format(len(self)))

    def __setitem__(self, index, value):
        self.imgs[index] = value[0]
        self.labels[index] = value[1]

    def __getitem__(self, index):

        fn = self.imgs[index]
        label = self.labels[index]
        img = fn

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):

        return len(self.imgs)



class CifarDataset(Dataset):

    def __init__(self, batch_num,df, mode='train', own_transform=None):
        batch_num += 1
        self.transform = own_transform
        df["label"] = df["class"].map(map_labels_cifar100)

        if mode == 'train':
            self.imgs = []
            self.labels = []
            temp = list(df["path"][(df["task"] == batch_num)&(df["type"] == "train")])
            labels = list(df["label"][(df["task"] == batch_num)&(df["type"] == "train")])


            self.imgs.extend([Image.open(x).convert('RGB') for x in temp])
            self.labels.extend(labels)
            print("  --> batch{}'-dataset consisting of {} samples".format(batch_num, len(self)))
            print(self.labels)
        else:
            self.imgs = []
            self.labels = []
            temp = list(df["path"][(df["task"] == batch_num) & (df["type"] == "test")])
            labels = list(df["label"][(df["task"] == batch_num) & (df["type"] == "test")])

            self.imgs.extend([Image.open(x).convert('RGB') for x in temp])
            self.labels.extend(labels)
            print("  --> test'-dataset consisting of {} samples".format(len(self)))

    def __setitem__(self, index, value):
        self.imgs[index] = value[0]
        self.labels[index] = value[1]

    def __getitem__(self, index):

        fn = self.imgs[index]
        label = self.labels[index]
        img = fn

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):

        return len(self.imgs)

class TrashDataset(Dataset):

    def __init__(self, batch_num, mode='train', own_transform=None, factor='trash_dist',base_path=r"C:\Users\ruiya\GitHub\lifelonglearning\datasets"):
        batch_num += 1
        self.transform = own_transform
        if mode == 'train':
            self.imgs = []
            self.labels = []
            for i,local_path in enumerate(trash_dist_path):
                temp = glob.glob(base_path + r'\{}\train\task{}\{}\*'.format(factor,batch_num, local_path))
                self.imgs.extend([Image.open(x).convert('RGB').resize((64, 64)) for x in temp])
                self.labels.extend([i] * len(temp))
            print("  --> batch{}'-dataset consisting of {} samples".format(batch_num, len(self)))
        else:
            self.imgs = []
            self.labels = []
            for i,local_path in enumerate(trash_dist_path):
                temp = glob.glob(base_path + '/{}/test/task{}/{}/*'.format(factor,batch_num,local_path))
                self.imgs.extend([Image.open(x).convert('RGB').resize((64, 64)) for x in temp])
                self.labels.extend([i] * len(temp))
            print("  --> test'-dataset consisting of {} samples".format(len(self)))

    def __setitem__(self, index, value):
        self.imgs[index] = value[0]
        self.labels[index] = value[1]

    def __getitem__(self, index):

        fn = self.imgs[index]
        label = self.labels[index]
        img = fn

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):

        return len(self.imgs)




class MyDataset(Dataset):

    def __init__(self, batch_num, mode='train', own_transform=None, factor='clutter'):
        batch_num += 1
        self.transform = own_transform
        if mode == 'train':
            self.imgs = []
            self.labels = []
            for i in range(len(datapath)):
                temp = glob.glob('img/' + factor + '/train/task{}/{}/*.jpg'.format(batch_num, datapath[i]))

                self.imgs.extend([Image.open(x).convert('RGB').resize((50, 50)) for x in temp])
                self.labels.extend([i] * len(temp))
            print("  --> batch{}'-dataset consisting of {} samples".format(batch_num, len(self)))

        else:
            self.imgs = []
            self.labels = []
            for i in range(len(datapath)):
                temp = glob.glob('img/' + factor + '/test/task{}/{}/*.jpg'.format(batch_num, datapath[i]))
                self.imgs.extend([Image.open(x).convert('RGB').resize((50, 50)) for x in temp])
                self.labels.extend([i] * len(temp))
            print("  --> test'-dataset consisting of {} samples".format(len(self)))

    def __setitem__(self, index, value):
        self.imgs[index] = value[0]
        self.labels[index] = value[1]

    def __getitem__(self, index):

        fn = self.imgs[index]
        label = self.labels[index]
        img = fn

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):

        return len(self.imgs)


# ----------------------------------------------------------------------------------------------------------#


class SubDataset(Dataset):

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "train_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.train_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.train_labels[index])
            elif hasattr(self.dataset, "test_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.test_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.test_labels[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


class ExemplarDataset(Dataset):

    def __init__(self, exemplar_sets, target_transform=None):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
        return (image, class_id_to_return)


def get_multitask_experiment(name, scenario, tasks, only_config=False, verbose=False,
                             exception=False, factor='clutter',df=None):
    if name == 'mydataset':
        classes_per_task = len(datapath)
        train_datasets = []
        test_datasets = []

        for i in range(tasks):
            train_datasets.append(MyDataset(i, mode='train', own_transform=my_transform, factor=factor))
            test_datasets.append(MyDataset(i, mode='test', own_transform=my_transform, factor=factor))
        config = {'size': 50, 'channels': 3, 'classes': len(datapath)}
    elif name == "taco_dataset":
        classes_per_task = 6
        train_datasets = []
        test_datasets = []

        for i in range(tasks):
            train_datasets.append(TacoDataset(i, mode='train', own_transform=my_transform, factor=factor))
            test_datasets.append(TacoDataset(i, mode='test', own_transform=my_transform, factor=factor))
        config = {'size': 50, 'channels': 3, 'classes': 6}
    elif name == "trash_dataset":
        classes_per_task = 5
        train_datasets = []
        test_datasets = []

        for i in range(tasks):
            train_datasets.append(TrashDataset(i, mode='train', own_transform=my_transform, factor=factor))
            test_datasets.append(TrashDataset(i, mode='test', own_transform=my_transform, factor=factor))
        config = {'size': 50, 'channels': 3, 'classes': 5}
    elif name == "trash_dist":
        classes_per_task = 5
        train_datasets = []
        test_datasets = []

        for i in range(tasks):
            train_datasets.append(TrashDataset(i, mode='train', own_transform=my_transform, factor=factor))
            test_datasets.append(TrashDataset(i, mode='test', own_transform=my_transform, factor=factor))
        config = {'size': 64, 'channels': 3, 'classes': 5}

    elif name == "trash_dist_class_imb":
        classes_per_task = 5
        train_datasets = []
        test_datasets = []

        for i in range(tasks):
            train_datasets.append(TrashDataset(i, mode='train', own_transform=my_transform, factor=factor))
            test_datasets.append(TrashDataset(i, mode='test', own_transform=my_transform, factor=factor))
        config = {'size': 64, 'channels': 3, 'classes': 5}

    elif name == "Pacs_dataset":
        classes_per_task = 7
        train_datasets = []
        test_datasets = []

        for i in range(tasks):
            train_datasets.append(PACSDataset(i, df=df,mode='train', own_transform=my_transform))
            test_datasets.append(PACSDataset(i, df=df,mode='test', own_transform=my_transform))
        config = {'size': 64, 'channels': 3, 'classes': 7}

    elif name == "cifar_dataset":
        classes_per_task = 20
        train_datasets = []
        test_datasets = []

        for i in range(tasks):
            train_datasets.append(CifarDataset(i, df=df,mode='train', own_transform=my_transform))
            test_datasets.append(CifarDataset(i, df=df,mode='test', own_transform=my_transform))
        config = {'size': 32, 'channels': 3, 'classes': 20}



    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)


