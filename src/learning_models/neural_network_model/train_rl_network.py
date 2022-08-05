import argparse
import math

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from learning_models.neural_network_model.rl_network import  RelationNetwork
from learning_models.neural_network_model.prototypical_network import CNN1DEncoder
from utils.gesture_data_related.read_data import read_data
from utils.neural_network_related.format_data_for_nn import format_batch_data
from utils.neural_network_related.task_generator import HandGestureTask, get_data_loader
from torch.utils.tensorboard import SummaryWriter

from torch.nn import functional as F


parser = argparse.ArgumentParser(description="One Shot Gesture Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 256)
parser.add_argument("-w","--class_num",type = int, default = 3)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 17)
parser.add_argument("-e","--episode",type = int, default= 1000) #initially : 1000000
parser.add_argument("-t","--test_episode", type = int, default = 10) # initially: 1000
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-u","--hidden_unit",type=int,default=8)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
HIDDEN_UNIT = args.hidden_unit


class train_rl_model:
    def __init__(self, seq_len = 30, learning_rate = LEARNING_RATE, train_episodes = EPISODE, test_episodes = TEST_EPISODE, embedding_dim = FEATURE_DIM):
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
        self.embedding_dim = embedding_dim
        self.train_episodes = train_episodes
        self.test_episodes = test_episodes
        self.learning_rate = learning_rate
        # At the time of train the network, this should be equal to all the classes present in the train dataset.
        self.num_classes = CLASS_NUM
        self.sample_num_per_class = SAMPLE_NUM_PER_CLASS
        self.batch_num_per_class = BATCH_NUM_PER_CLASS
        self.seq_len = seq_len
        self.feature_encoder = CNN1DEncoder(seq_len=self.seq_len, feature_dim=self.embedding_dim)
        print(self.feature_encoder)
        self.relation_network = RelationNetwork(seq_len=self.seq_len, hidden_size=HIDDEN_UNIT, feature_dim=self.embedding_dim)
        print(self.relation_network)
        self.feature_encoder.apply(self._weights_init)
        self.relation_network.apply(self._weights_init)
        self.feature_encoder , self.relation_network = self.feature_encoder.to(self.device), self.relation_network.to(self.device)
        self.feature_encoder_optim = torch.optim.Adam(self.feature_encoder.parameters(), lr= self.learning_rate)
        self.feature_encoder_scheduler = StepLR(self.feature_encoder_optim, step_size=EPISODE/10, gamma=0.1)
        self.relation_network_optim = torch.optim.Adam(self.relation_network.parameters(), lr=self.learning_rate)
        self.relation_network_scheduler = StepLR(self.relation_network_optim, step_size=EPISODE/10, gamma=0.1)
        self.writer = SummaryWriter()


        # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.



    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # print(m.kernel_size)
            n = m.kernel_size[0] * m.out_channels
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
        training_data_path = './../../../HandDataset/train_sim_red'
        testing_data_path = './../../../HandDataset/train_sim_red'
        data_path = ""
        if(train):
            data_path = training_data_path
        else:
            data_path = testing_data_path

        data_list, _ = read_data(path=data_path, window_size=30)
        total_num_classes, data_dict = format_batch_data(data_list)
        data_dict["labels"] = data_dict["labels"] - 1
        task = HandGestureTask(data_dict=data_dict, req_num_classes=self.num_classes,
                               train_num=self.sample_num_per_class, test_num=self.batch_num_per_class)


        trainDataLoader = get_data_loader(task=task, num_inst=self.sample_num_per_class,
                                          num_classes=self.num_classes,
                                          split='train')
        testDataLoader = get_data_loader(task=task, num_inst=self.batch_num_per_class,
                                         num_classes=self.num_classes,
                                         split='test')
        return trainDataLoader, testDataLoader

# debugging notes: couple with sample data rather than batch; auto-encoders; increasing the dataset; prune the relation network

    def _predict_correlation(self, samples, batches):
        # calculate embedding for support set and query set.
        sample_features = self.feature_encoder(
            Variable(samples).to(self.device, dtype=torch.float))  # (req_num_classes*num_inst_class_samples) x feature_dim x output_seq_len
        sample_features = sample_features.view(self.num_classes, self.sample_num_per_class, self.embedding_dim,
                                               -1)  # req_num_classes x num_inst_class x feature_dim x output_seq_len
        # All instances of a particular clss are added.
        # Assumption: All the instances of a particular class are grouped together.
        sample_features = torch.sum(sample_features, 1).squeeze(1)  # num_classes x feature_dim x output_seq_len
        batch_features = self.feature_encoder(
            Variable(batches).to(self.device, dtype=torch.float))  # inst_per_class_test*req_num_classes x feature_dim x output_seq_len

        # calculate relations
        sample_features_ext = sample_features.unsqueeze(0).repeat(self.batch_num_per_class * self.num_classes, 1, 1, 1)
        # dimension sample_features_ext = inst_per_class_test*req_num_classes x req_num_classes x 64 x 30
        batch_features_ext = batch_features.unsqueeze(0).repeat(self.num_classes, 1, 1, 1)
        # dimension batch_features_ext = req_num_classes x inst_per_class_test*req_num_classes x 64 x 30
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        # dimension batch_features_ext_transpose = inst_per_class_test*req_num_classes x req_num_classes x 64 x 30

        # This would create all_possible pairs of support_set and query_set.
        # dimension = inst_per_class_test*req_num_classes*req_num_classes(num of data in support set) x (64+64) x 30
        relation_pairs = torch.cat((batch_features_ext, sample_features_ext), 2).view(-1, self.embedding_dim * 2, sample_features.shape[2])
        # dimension(relations) = (req_num_classes*req_num_classes*inst_per_class_test) x 1   ->  req_num_classes*inst_per_class_test x req_num_classes
        # relation_pairs_debug = torch.cat((batch_features, batch_features), 1) # inst_per_class_test*req_num_classes x 2*feature_dim x
        relations = self.relation_network(relation_pairs).view(-1, self.num_classes)
        # relations = relations
        # relations = self.relation_network(relation_pairs_debug)
        return relations


    def train_model(self):
        print("train the model")
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
            # This method assumes that train set has labels starting from 0 to (num_classes-1)
            one_hot_labels = Variable(
                torch.zeros(self.batch_num_per_class * self.num_classes, self.num_classes).scatter_(1, batch_labels.view(-1, 1), 1)).to(self.device)
            # one_hot_labels_debug = Variable(
            #     torch.ones(relations.shape[0])).to(self.device)
            log_p_y = F.log_softmax(relations, dim=1)
            loss_correct = (-1 * log_p_y * one_hot_labels).sum(1).squeeze().mean()
            loss = mse(relations, one_hot_labels)
            # train
            self.feature_encoder.zero_grad()
            self.relation_network.zero_grad()

            loss.backward()

            # torch.nn.utils.clip_grad_norm(self.feature_encoder.parameters(), 2)
            # torch.nn.utils.clip_grad_norm(self.relation_network.parameters(), 2)

            self.feature_encoder_optim.step()
            self.relation_network_optim.step()

            # DEBUG: Logging in the parameters of the network

            for name, weight in self.feature_encoder.named_parameters():
                self.writer.add_histogram(name, weight, episode)
                self.writer.add_histogram(f'{name}.grad', weight.grad, episode)

            for name, weight in self.relation_network.named_parameters():
                self.writer.add_histogram(name, weight, episode)
                self.writer.add_histogram(f'{name}.grad', weight.grad, episode)

            if (episode + 1) % 2 == 0:
                print("episode______________________________________________:", episode + 1, "loss", loss.data)
                # print("sample_labels: ",sample_labels)
                print("batch_labels: ",batch_labels)
                # print("batch_labels_one_hot: ", one_hot_labels)
                print("relations obtained: ", relations)
                _, predict_labels = torch.max(relations.data, 1)
                print("predicted labels:", predict_labels)
                # rewards = [1 if predict_labels[j] == batch_labels[j] else 0 for j in
                #            range(self.num_classes * self.batch_num_per_class)]

                rewards = [1 if predict_labels[j] == batch_labels[j] else 0 for j in
                           range(self.num_classes * self.batch_num_per_class)]

                total_rewards = np.sum(rewards)
                train_accuracy = total_rewards / (self.num_classes*self.batch_num_per_class)
                print( "training accuracy", train_accuracy)
                # self.writer.add_scalar("Loss/train", loss.data, episode+1)

            if (episode + 1) % 50 == 0:
                # test
                self.relation_network.eval()
                self.feature_encoder.eval()
                print("Testing...")
                total_rewards = 0
                for i in range(self.test_episodes):
                    sample_dataloader, test_dataloader = self._load_data(train=False)
                    sample_images, sample_labels = sample_dataloader.__iter__().next()
                    test_images, test_labels = test_dataloader.__iter__().next()
                    relations = self._predict_correlation(sample_images, test_images)
                    _, predict_labels = torch.max(relations.data, 1)

                    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in
                               range(self.num_classes * self.batch_num_per_class)]

                    total_rewards += np.sum(rewards)

                test_accuracy = total_rewards / (self.num_classes*self.batch_num_per_class*self.test_episodes)
                self.relation_network.train()
                self.feature_encoder.train()
                print("test accuracy:", test_accuracy)

                if test_accuracy > last_accuracy:
                    # save networks
                    torch.save(self.feature_encoder.state_dict(),
                               str("./../../saved_models/hand_feature_encoder_" + str(self.num_classes) + "way_" + str(
                                   self.sample_num_per_class) + "shot.pkl"))
                    torch.save(self.relation_network.state_dict(),
                               str("./../../saved_models/hand_relation_network_" + str(self.num_classes) + "way_" + str(
                                   self.sample_num_per_class) + "shot.pkl"))

                    print("save networks for episode:", episode)

                    last_accuracy = test_accuracy




network_trainer = train_rl_model()
network_trainer.train_model()