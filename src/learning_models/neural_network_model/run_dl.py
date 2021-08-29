import torch

from src.utils.neural_network_related import format_data_for_nn


class DL_run:
    def __init__(self, path_to_data, folder_name, window_size):
        self.val_loader = None
        self.train_loader = None
        self.batch_size = 16
        self.test_batch_size = 1
        self.val_batch_size = 16
        self.device = format_data_for_nn.get_device()
        self.path_to_data = path_to_data
        self.folder_name = folder_name
        self.window_size = window_size
        # device = "cpu"

    def setupDL(self, obj, output_size):
        self.train_loader, self.val_loader, self.test_loader, seq_len = format_data_for_nn.get_data_for_training(
            batch_size=self.batch_size,
            val_batch_size=self.val_batch_size,
            test_batch_size=self.test_batch_size,
            path_to_data=self.path_to_data,
            folder_name=self.folder_name,
            window_size=self.window_size,
        )
        self.model = obj.CNN1D(seq_len, self.device, output_size=output_size).to(self.device)

    def trainDL(self, obj, model_path, lr=0.002, epochs=100):
        self.nn_train = obj.train_neural_network(model=self.model, device=self.device,
                                                 lr=lr, epochs=epochs, train_loader=self.train_loader,
                                                 test_loader=self.test_loader,
                                                 val_loader=self.val_loader)
        self.nn_train.train_model(model_save_path=model_path)

    def evalDL(self, obj, model_path):
        self.model.load_state_dict(torch.load(model_path))  # map_location=torch.device("cpu")
        self.nn_train.evaluate_model(self.test_batch_size)


if __name__ == '__main__':
    run = DL_run()

    if False:
        print("RUNNING CNN1D")
        run.setupDL(CNN1D, path_to_data, folder_name)
        run.trainDL(CNN1D, lr=0.002, epochs=800)
        run.evalDL(CNN1D)

    if False:
        print("RUNNING CNN_GRU")
        run.setupDL(CNN_GRU, path_to_data, folder_name)
        run.trainDL(CNN_GRU, lr=0.002, epochs=100)
        run.evalDL(CNN_GRU)

    if False:
        print("RUNNING GRU")
        run.setupDL(GRU)
        run.trainDL(GRU, lr=0.005, epochs=100)
        run.evalDL()
