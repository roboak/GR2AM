import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import random
from src.utils.gesture_data_related.read_data import read_data
from src.utils.neural_network_related.format_data_for_nn import format_batch_data

# This class is for a single task generation for both meta train and meta testing.
# For meta train, we use all 20 samples without valid set (empty here).
# For meta testing, we use 1 or 5 shot samples for train, while using the same number of samples for validation.
# If set num_samples = 20 and chracter_folders = metatrain_character_folders, we generate tasks for meta train
# If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing

class HandGestureTask(object):
    def __init__(self, data_dict, req_num_classes, train_num, test_num):
        self.req_num_classes= req_num_classes
        # get indices of samples for a particular class. For example, if label = [1,2,1,1,2,3]
        # dict_label = {'1'= [0, 2, 3]; '2' = [1, 4]; 3= [5]}
        label_indices_dict = self.get_pos_for_each_label(data_dict["labels"])

        # Out of all the classes in the data folder, randomly select the 'req_num_classes'
        randomly_sampled_classes = random.sample(set(data_dict["labels"].tolist()), req_num_classes)
        # for each class some samples have to be used for train(either during train or in support set)
        # and some samples have to be used for testing.

        # Each cell corresponds to a class. It is assumed
        train_indices = []
        test_indices = []
        for i in randomly_sampled_classes:
            # for every class select randomly the 'train_num' of indices which will be used for train
            train_indices_per_class = random.sample(label_indices_dict[i], train_num)
            #  for every class select randomly the 'test_num' of indices which will be used for testing. These indices should be disjoint with the
            # indices selected for train for that particular class.
            test_indices_per_class = random.sample(set(label_indices_dict[i]).difference(set(train_indices_per_class)), test_num)
            train_indices += train_indices_per_class
            test_indices += test_indices_per_class


        self.train_roots, self.train_labels = data_dict["data"][train_indices], data_dict["labels"][train_indices]
        self.test_roots, self.test_labels = data_dict["data"][test_indices], data_dict["labels"][test_indices]

        print("train_indices: ", train_indices)
        print("train_labels: ", self.train_labels)

        print("test_indices ", test_indices)
        print("test_labels: ", self.test_labels)
        # self.train_roots, self.train_labels = self.unison_shuffled_copies(data_dict["data"][train_indices], data_dict["labels"][train_indices])
        # self.test_roots, self.test_labels = self.unison_shuffled_copies(data_dict["data"][test_indices], data_dict["labels"][test_indices])


    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def get_pos_for_each_label(self, labels) -> dict:
        label_indices_dict = dict()
        for index, ele in enumerate(labels):
            if ele in label_indices_dict.keys():
                temp = label_indices_dict[ele]
                temp.append(index)
                label_indices_dict[ele] = temp
            else:
                label_indices_dict[ele] = [index]
        return label_indices_dict


class HandGestureDataSet(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.data_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.data_roots)

    def __getitem__(self, idx):
        data = self.data_roots[idx]
        label = self.labels[idx]
        return data, label

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1
def debug():
    data_list, _ = read_data(path='./../../../HandDataset/Abdul_New_Data', window_size=30)
    total_num_classes, data_dict = format_batch_data(data_list)
    task = HandGestureTask(data_dict = data_dict, req_num_classes=5,
                           train_num=1, test_num=5)
    train_hand_dataset = HandGestureDataSet(task=task, split='train') #dimensions = batch_size(req_num_classes) x seq_len(30) x num_channels(63)
    test_hand_dataset = HandGestureDataSet(task=task, split='test')

def get_data_loader(task, num_inst, num_classes, split='train'):

    dataset = HandGestureDataSet(task=task, split=split)
    loader = DataLoader(dataset, batch_size=num_inst*num_classes, shuffle= False)
    return loader


def debug():
    data_list, _ = read_data(path='./../../../HandDataset/Abdul_New_Data', window_size=30)
    req_num_classes = 5
    inst_per_class_train = 5
    inst_per_class_test = 2
    total_num_classes, data_dict = format_batch_data(data_list)
    task = HandGestureTask(data_dict = data_dict, req_num_classes=req_num_classes,
                           train_num=inst_per_class_train, test_num=inst_per_class_test)
    trainDataLoader = get_data_loader(task=task, num_inst=inst_per_class_train, num_classes=req_num_classes, split='train')


debug()

