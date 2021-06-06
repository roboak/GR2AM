import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset



def format_data(dataset):  # read data in the format of [total_data_size, sequence length, feature_size, feature_dim]
    seq_len = dataset[0].data.shape[0]
    num_features = dataset[0].data.shape[1]
    feature_dim = dataset[0].data.shape[2]
    data_array = np.zeros((len(dataset), seq_len, num_features, feature_dim))
    labels = np.zeros(len(dataset))
    data_dict = {}
    for idx, data in enumerate(dataset):
        data_array[idx] = data.data
        labels[idx] = data.label
    # Return all data stacked together
    # Shape of data changes from [batch_size,seq_len,input_dim,feature_dims] -> [batch_size,seq_len,input_dim*feature_dims]
    data_dict["data"] = data_array.reshape(len(data_array), seq_len, num_features * feature_dim).astype(np.float)
    data_dict["labels"] = labels.astype(int)
    num_classes = len(np.unique(data_dict["labels"]))
    return num_classes, data_dict


def hot_encoding(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y - 1]


def to_categorical(labels, num_classes):
    new_labels = np.zeros((len(labels), num_classes))
    for idx, label in enumerate(labels):
        new_labels[idx] = hot_encoding(label, num_classes)
    return new_labels


def split_training_test_valid(data_dict, num_labels):
    data_dict["labels"] = data_dict["labels"] - 1
    #data_dict["labels"] = to_categorical(data_dict["labels"], num_labels)
    X_train, X_test, y_train, y_test = train_test_split(data_dict["data"], data_dict["labels"], test_size=0.3,
                                                        random_state=42)
    split_frac = 0.5
    split_id = int(split_frac * len(X_test))
    X_val, X_test = X_test[:split_id], X_test[split_id:]
    y_val, y_test = y_test[:split_id], y_test[split_id:]
    # print(y_train, y_test)
    return X_train, X_test, X_val, y_train, y_test, y_val


def get_mini_batches(X_train, X_test, X_val, y_train, y_test, y_val, batch_size, test_batch_size):
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=test_batch_size, drop_last=True)
    return train_loader, val_loader, test_loader


def get_device():
    is_cuda = torch.cuda.is_available()
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    return device
