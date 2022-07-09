import math

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from learning_models.neural_network_model.rl_network import CNN1DEncoder, RelationNetwork
from utils.gesture_data_related.read_data import read_data
from utils.neural_network_related.format_data_for_nn import format_batch_data
from utils.neural_network_related.task_generator import HandGestureTask, get_data_loader
from torch.utils.tensorboard import SummaryWriter

class train_rl_model:
    def __init__(self, seq_len = 30, learning_rate = 0.001, train_episodes = 1000000, test_episodes = 1000):
        self.train_episodes = train_episodes
        self.test_episodes = test_episodes
        self.learning_rate = learning_rate
        # At the time of training the network, this should be equal to all the classes present in the training dataset.
        self.num_classes = 5
        self.sample_num_per_class = 5
        self.batch_num_per_class = 10
        self.seq_len = seq_len
        self.feature_encoder = CNN1DEncoder(seq_len=30).to(self.device)
        self.relation_network = RelationNetwork(seq_len=30, hidden_size=10).to(self.device)
        self.feature_encoder.apply(self._weights_init)
        self.relation_network.apply(self._weights_init)
        self.feature_encoder_optim = torch.optim.Adam(self.feature_encoder.parameters(), lr= self.learning_rate)
        self.feature_encoder_scheduler = StepLR(self.feature_encoder_optim, step_size=100000, gamma=0.5)
        self.relation_network_optim = torch.optim.Adam(self.relation_network.parameters(), lr=self.learning_rate)
        self.relation_network_scheduler = StepLR(self.relation_network_optim, step_size=100000, gamma=0.5)
        self.writer = SummaryWriter()


        # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
        def get_device():
            is_cuda = torch.cuda.is_available()
            device = ""
            if is_cuda:
                device = torch.device("cuda")
                print("GPU is available")
            else:
                device = torch.device("cpu")
                print("GPU not available, CPU used")
            return device
        self.device = get_device()


    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            print(m.kernel_size)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())

    def _load_data(self, train = True):
        training_data_path = './../../HandDataset/training'
        testing_data_path = './../../HandDataset/test'
        data_path = ""
        test_num = 0
        if(train):
            data_path = training_data_path
            test_num = self.batch_num_per_class
        else:
            data_path = testing_data_path
            test_num = self.sample_num_per_class

        data_list, _ = read_data(path=data_path, window_size=30)
        total_num_classes, data_dict = format_batch_data(data_list)

        task = HandGestureTask(data_dict=data_dict, req_num_classes=self.num_classes,
                               train_num=self.sample_num_per_class, test_num=test_num)

        trainDataLoader = get_data_loader(task=task, num_inst=self.sample_num_per_class,
                                          num_classes=self.num_classes,
                                          split='train')
        testDataLoader = get_data_loader(task=task, num_inst=test_num,
                                         num_classes=self.num_classes,
                                         split='test')
        return trainDataLoader, testDataLoader

    def _predict_correlation(self, samples, batches):
        # calculate embedding for support set and query set.
        sample_features = self.feature_encoder(
            Variable(samples).to(self.device))  # (req_num_classes*num_inst_class)x64x30
        sample_features = sample_features.view(self.num_classes, self.sample_num_per_class, 64,
                                               30)  # req_num_classes x num_inst_class x 64 x 30
        # All instances of a particular clss are added.
        # Assumption: All the instaces of a particular class are grouped together.
        sample_features = torch.sum(sample_features, 1).squeeze(1)  # req_num_classes x 64 x 30
        batch_features = self.feature_encoder(
            Variable(batches).to(self.device))  # inst_per_class_test*req_num_classes x 64 x (seq_len=30)

        # calculate relations
        sample_features_ext = sample_features.unsqueeze(0).repeat(self.batch_num_per_class * self.num_classes, 1, 1, 1)
        # dimension sample_features_ext = inst_per_class_test*req_num_classes x req_num_classes x 64 x 30
        batch_features_ext = batch_features.unsqueeze(0).repeat(self.num_classes, 1, 1, 1)
        # dimension batch_features_ext = req_num_classes x inst_per_class_test*req_num_classes x 64 x 30
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        # dimension batch_features_ext_transpose = inst_per_class_test*req_num_classes x req_num_classes x 64 x 30

        # This would create all_possible pairs of support_set and query_set.
        # dimension = inst_per_class_test*req_num_classes*req_num_classes(num of data in support set) x (64+64) x 30
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, 64 * 2, 30)

        # dimension(relations) = (req_num_classes*inst_per_class_test) x 1   ->  inst_per_class_test x req_num_classes
        relations = self.relation_network(relation_pairs).view(-1, self.num_classes)
        return relations


    def train_model(self):
        print("training the model")
        last_accuracy = 0.0
        for episode in range(self.train_episodes):
            self.feature_encoder_scheduler.step(episode)
            self.relation_network_scheduler.step(episode)

            # init dataset
            # sample_dataloader is used for support set
            # batch_dataloader is used for query set
            sample_dataloader, batch_dataloader = self._load_data()
            # sample datas
            samples, sample_labels = sample_dataloader.__iter__().next()
            batches, batch_labels = batch_dataloader.__iter__().next()
            relations = self._predict_correlation(samples, batches)
            mse = nn.MSELoss().to(self.device)
            # This method assumes that training set has labels starting from 0 to (num_classes-1)
            one_hot_labels = Variable(
                torch.zeros(self.batch_num_per_class * self.num_classes, self.num_classes).scatter_(1, batch_labels.view(-1, 1), 1)).to(self.device)
            loss = mse(relations, one_hot_labels)

            # training
            self.feature_encoder.zero_grad()
            self.relation_network.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm(self.feature_encoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm(self.relation_network.parameters(), 0.5)

            self.feature_encoder_optim.step()
            self.relation_network_optim.step()

            if (episode + 1) % 100 == 0:
                print("episode:", episode + 1, "loss", loss.data[0])
                self.writer.add_scalar("Loss/train", loss.data[0], episode+1)

            if (episode + 1) % 5000 == 0:
                # test
                print("Testing...")
                total_rewards = 0
                for i in range(self.test_episodes):
                    sample_dataloader, test_dataloader = self._load_data(train=False)
                    sample_images, sample_labels = sample_dataloader.__iter__().next()
                    test_images, test_labels = test_dataloader.__iter__().next()
                    relations = self._predict_correlation(sample_images, test_images)
                    _, predict_labels = torch.max(relations.data, 1)

                    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in
                               range(self.num_classes * self.sample_num_per_class)]

                    total_rewards += np.sum(rewards)

                test_accuracy = total_rewards / 1.0 / self.num_classes / self.sample_num_per_class / self.test_episodes

                print("test accuracy:", test_accuracy)

                if test_accuracy > last_accuracy:
                    # save networks
                    torch.save(self.feature_encoder.state_dict(),
                               str("./models/hand_feature_encoder_" + str(self.num_classes) + "way_" + str(
                                   self.sample_num_per_class) + "shot.pkl"))
                    torch.save(self.relation_network.state_dict(),
                               str("./models/hand_relation_network_" + str(self.num_classes) + "way_" + str(
                                   self.sample_num_per_class) + "shot.pkl"))

                    print("save networks for episode:", episode)

                    last_accuracy = test_accuracy


