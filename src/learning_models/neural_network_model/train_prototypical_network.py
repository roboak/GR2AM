import argparse
import math

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from learning_models.neural_network_model.prototypical_network import CNN1DEncoder
from utils.gesture_data_related.read_data import read_data
from utils.neural_network_related.format_data_for_nn import format_batch_data
from utils.neural_network_related.task_generator import HandGestureTask, get_data_loader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F


parser = argparse.ArgumentParser(description="One Shot Gesture Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-w","--class_num",type = int, default = 3)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 17)
parser.add_argument("-e","--episode",type = int, default= 1500) #initially : 1000000
parser.add_argument("-t","--test_episode", type = int, default = 10) # initially: 1000
parser.add_argument("-l","--learning_rate", type = float, default = 0.005)
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


class train_proto_model:
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
        # self.feature_encoder.apply(self._weights_init)
        # self.relation_network.apply(self._weights_init)
        self.feature_encoder  = self.feature_encoder.to(self.device)
        self.feature_encoder_optim = torch.optim.Adam(self.feature_encoder.parameters(), lr= self.learning_rate)
        self.feature_encoder_scheduler = StepLR(self.feature_encoder_optim, step_size=EPISODE/10, gamma=0.5)
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
        test_num = self.batch_num_per_class
        if(train):
            data_path = training_data_path
        else:
            data_path = testing_data_path

        data_list, _ = read_data(path=data_path, window_size=30)
        total_num_classes, data_dict = format_batch_data(data_list)
        data_dict["labels"] = data_dict["labels"] - 1
        task = HandGestureTask(data_dict=data_dict, req_num_classes=self.num_classes,
                               train_num=self.sample_num_per_class, test_num=test_num)


        trainDataLoader = get_data_loader(task=task, num_inst=self.sample_num_per_class,
                                          num_classes=self.num_classes,
                                          split='train')
        testDataLoader = get_data_loader(task=task, num_inst=test_num,
                                         num_classes=self.num_classes,
                                         split='test')
        return trainDataLoader, testDataLoader

# debugging notes: couple with sample data rather than batch; auto-encoders; increasing the dataset; prune the relation network
    def _euclidean_dist(self, x, y):
        '''
        Compute euclidean distance between two tensors
        '''
        # x: num_class*num_class*num_inst_per_class_test x feature_dim * seq_len
        # y: num_class*num_class*num_inst_per_class_test x feature_dim * seq_len
        return torch.sqrt(torch.pow(x - y, 2).sum(1))

    def _cosine_dist(self, x, y):
        '''
        Compute euclidean distance between two tensors
        '''
        # x: num_class*num_class*num_inst_per_class_test x feature_dim * seq_len
        # y: num_class*num_class*num_inst_per_class_test x feature_dim * seq_len
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(x,y)

    # def _calculate_correlation(self, x, y):

    def _forward_pass(self, samples, batches, batch_labels):
        # calculate embedding for support set and query set.
        sample_features = self.feature_encoder(
            Variable(samples).to(self.device, dtype=torch.float))  # (req_num_classes*num_inst_class)x64x30
        sample_features = sample_features.view(self.num_classes, self.sample_num_per_class, self.embedding_dim,
                                               -1)  # req_num_classes x num_inst_class x 64 x 30
        # All instances of a particular clss are added.
        # Assumption: All the instances of a particular class are grouped together. The following line of code
        # calculates the average of all the samples of a particular class.
        sample_features = torch.sum(sample_features, 1).squeeze(1) / self.sample_num_per_class # req_num_classes x 64 x 30
        batch_features = self.feature_encoder(
            Variable(batches).to(self.device, dtype=torch.float))  # inst_per_class_test*req_num_classes x 64 x (seq_len=30)

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
        #dists dimension = inst_per_class_test*req_num_classes*req_num_classes x 1

        dists_cosine = self._cosine_dist(relation_pairs[:, 0:self.embedding_dim, :].view(self.num_classes*self.num_classes*self.batch_num_per_class, -1),
                                     relation_pairs[:, self.embedding_dim:, :].view(self.num_classes*self.num_classes*self.batch_num_per_class, -1))
        #dists dimension = inst_per_class_test*req_num_classes x req_num_classes
        dists_cosine = dists_cosine.view(-1, self.num_classes)

        dists_euc = self._euclidean_dist(relation_pairs[:, 0:self.embedding_dim, :].view(self.num_classes*self.num_classes*self.batch_num_per_class, -1),
                                     relation_pairs[:, self.embedding_dim:, :].view(self.num_classes*self.num_classes*self.batch_num_per_class, -1))
        #dists dimension = inst_per_class_test*req_num_classes x req_num_classes
        dists_euc = dists_euc.view(-1, self.num_classes)

        dists =  dists_euc #- dists_cosine
        norm_dists = (dists - dists.min(1, True).values)/(dists.max(1, True).values - dists.min(1, True).values)
        # print("dists:", dists)
        log_p_y = F.log_softmax(-1*dists, dim=1)
        _, predicted_label = norm_dists.min(1)
        acc = torch.eq(predicted_label, batch_labels.to(self.device)).float().mean()
        one_hot_labels = torch.zeros(self.batch_num_per_class * self.num_classes, self.num_classes).scatter_(1, batch_labels.view(-1, 1), 1).to(self.device)
        one_hot_inv_labels = torch.ones(self.batch_num_per_class * self.num_classes, self.num_classes).scatter_(1, batch_labels.view(-1, 1), 0).to(self.device)
        # reg = (dists.sum(1) - (dists*one_hot_labels).sum(1)).sum().squeeze()
        # reg = 1/reg
        lam = 0.3
        loss_correct = (-1 * log_p_y * one_hot_labels).sum(1).squeeze().mean()
        mse = nn.MSELoss().to(self.device)
        loss_mse = mse(norm_dists, one_hot_inv_labels)
        loss = loss_mse #+ 0.01*loss_correct
        return loss, acc, predicted_label, F.softmax(-1*dists, dim = 1)


    def train_model(self):
        print("train the model")
        last_accuracy = 0.0
        for episode in range(self.train_episodes):
            self.feature_encoder_scheduler.step(episode)

            # init dataset
            # sample_dataloader is used for support set
            # batch_dataloader is used for query set
            sample_dataloader, batch_dataloader = self._load_data()
            # sample datas
            samples, sample_labels = sample_dataloader.__iter__().next()
            batches, batch_labels = batch_dataloader.__iter__().next()
            loss, acc, predict_labels, prob = self._forward_pass(samples, batches, batch_labels)

            # train
            self.feature_encoder.zero_grad()
            loss.backward()
            self.writer.add_scalar("Training/loss", loss.data, episode + 1)
            self.writer.add_scalar("Training/accuracy", acc.data, episode + 1)

            # torch.nn.utils.clip_grad_norm(self.feature_encoder.parameters(), 2)
            # torch.nn.utils.clip_grad_norm(self.relation_network.parameters(), 2)

            self.feature_encoder_optim.step()

            # DEBUG: Logging in the parameters of the network

            for name, weight in self.feature_encoder.named_parameters():
                self.writer.add_histogram(name, weight, episode)
                self.writer.add_histogram(f'{name}.grad', weight.grad, episode)


            if (episode + 1) % 5 == 0:
                print("episode_________________________:", episode + 1, "loss", loss.data)
                # print("sample_labels: ",sample_labels)
                print("batch_labels: ",batch_labels)
                # print("batch_labels_one_hot: ", one_hot_labels)
                # print("relations obtained: ", relations)
                # _, predict_labels = torch.max(relations.data, 1)
                print("predicted labels:", predict_labels)
                # rewards = [1 if predict_labels[j] == batch_labels[j] else 0 for j in
                #            range(self.num_classes * self.batch_num_per_class)]
                # total_rewards = np.sum(rewards)
                # train_accuracy = total_rewards / (self.num_classes*self.batch_num_per_class)
                print("training accuracy", acc)
                print("probabilities: ", prob)
                # self.writer.add_scalar("Loss/train", loss.data, episode+1)

            if (episode + 1) % 20 == 0:
                # test
                print("Testing...")
                test_accuracy = 0
                self.feature_encoder.eval()
                for i in range(self.test_episodes):
                    sample_dataloader, test_dataloader = self._load_data(train=False)
                    sample_images, sample_labels = sample_dataloader.__iter__().next()
                    test_images, test_labels = test_dataloader.__iter__().next()
                    loss, acc, predict_labels,_ = self._forward_pass(sample_images, test_images, test_labels)
                    # relations = self._forward_pass(sample_images, test_images)
                    # _, predict_labels = torch.max(relations.data, 1)
                    #
                    # rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in
                    #            range(self.num_classes * self.batch_num_per_class)]
                    #
                    # total_rewards += np.sum(rewards)
                    test_accuracy += acc
                self.feature_encoder.train()
                test_accuracy = test_accuracy / self.test_episodes

                print("test accuracy:", test_accuracy)
                self.writer.add_scalar("Testing/acc", test_accuracy.data, (episode + 1) % 20)

                if test_accuracy > last_accuracy:
                    # save networks
                    torch.save(self.feature_encoder.state_dict(),
                               str("./../../saved_models/hand_feature_encoder_" + str(self.num_classes) + "way_" + str(
                                   self.sample_num_per_class) + "shot.pkl"))
                    # torch.save(self.relation_network.state_dict(),
                    #            str("./../../saved_models/hand_relation_network_" + str(self.num_classes) + "way_" + str(
                    #                self.sample_num_per_class) + "shot.pkl"))

                    print("save networks for episode:", episode)

                    last_accuracy = test_accuracy


network_trainer = train_proto_model()
network_trainer.train_model()