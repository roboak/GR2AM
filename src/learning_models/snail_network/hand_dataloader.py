from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from learning_models.snail_network.batch_sampler import BatchSampler
from utils.gesture_data_related.read_data import read_data
from utils.neural_network_related.format_data_for_nn import format_batch_data


class HandGestureDataSet(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        return len(self.data_dict["labels"])

    def __getitem__(self, idx):
        data = self.data_dict["data"][idx]
        label = self.data_dict["labels"][idx]
        return data, label

def get_dataloader(data_path, sample_num_per_class, num_iterations, batch_size, classes_per_it = None):
    data_list, _ = read_data(path=data_path, window_size=30)
    total_num_classes, data_dict = format_batch_data(data_list)
    if classes_per_it == None:
        classes_per_it = total_num_classes
    handGestureDataset = HandGestureDataSet(data_dict)
    sampler = BatchSampler(labels=data_dict["labels"],
                                          classes_per_it=classes_per_it,
                                          num_samples=sample_num_per_class,
                                          iterations=num_iterations,
                                          batch_size=batch_size)
    dataloader = DataLoader(handGestureDataset, sampler= sampler)
    return dataloader

def debug():
    dataloader = get_dataloader(data_path='./../../../HandDataset/train', sample_num_per_class= 5, num_iterations=2, batch_size=2)
    for epoch in range(2):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(dataloader)
        for batch in tqdm(tr_iter):
            x, y = batch


# debug()
