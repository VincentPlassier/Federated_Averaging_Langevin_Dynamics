import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


class FedDataset:

    def __init__(self, dataset_name, path_dataset, transform = None, max_data_size = np.inf):
        self.transform = transform
        # Define the parameter of the dataset
        params_train = {"root": path_dataset, "train": True, "transform": self.transform, "download": True}
        params_test = {"root": path_dataset, "train": False, "transform": self.transform}

        # Load the function associated with the chosen dataset
        dataset = getattr(torchvision.datasets, dataset_name)

        # Define the datasets
        trainset = dataset(**params_train)
        lenght = len(trainset)
        # We only consider a subset of the dataset if max_data_size < len(trainset)
        if max_data_size < lenght:
            trainset = torch.utils.data.Subset(trainset, np.random.choice(lenght, max_data_size, replace=False))

        # Modify the shape of the data to save time during the training stage
        self.inputs = trainset.dataset.data if max_data_size < lenght else trainset.data
        if torch.is_tensor(self.inputs):
            self.inputs = self.inputs.numpy()
        self.targets = trainset.dataset.targets if max_data_size < lenght else trainset.targets

        # Define the testloader
        batch_size_test = 500
        testset = dataset(**params_test)
        self.testloader = DataLoader(testset, batch_size_test, shuffle=False)

    def create_clients_loader(self, num_clients, batch_size = 64, hard = True):
        datasets_dict = {}
        for label in np.unique(self.targets):
            idx_label = np.where(self.targets == label)[0]
            p = np.random.uniform(0, 1, size=num_clients) if hard else np.ones(num_clients)
            num_data = (1 + np.round((len(idx_label) - 1 * num_clients) * np.random.dirichlet(p))).astype('int')
            num_total = 0
            for client, num in enumerate(num_data):
                if client == num_clients - 1:
                    num = len(idx_label) - num_total
                idx_client = idx_label[num_total: num_total + num]
                if num == 0:
                    continue
                if client not in datasets_dict.keys():
                    datasets_dict[client] = [np.take(self.inputs, idx_client, axis=0),
                                             np.take(self.targets, idx_client, axis=0)]
                else:
                    datasets_dict[client][0] = np.vstack(
                        (datasets_dict[client][0], np.take(self.inputs, idx_client, axis=0)))
                    datasets_dict[client][1] = np.hstack(
                        (datasets_dict[client][1], np.take(self.targets, idx_client, axis=0)))
                num_total += num

        datasets = []
        for x, y in datasets_dict.values():
            if self.transform is not None:
                x_transform = self.transform(x.transpose(1, 2, 0))[:, np.newaxis] if x.ndim == 3 else torch.stack(
                    list(map(transform, x)))
                datasets.append([x_transform, torch.from_numpy(y)])
            else:
                datasets.append([x, y])
        return [DataLoader(TensorDataset(X, Y), batch_size, shuffle=True) for X, Y in datasets]


if __name__ == '__main__':
    # Path to the dataset
    path_dataset = './'

    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Define some parameters
    dataset_name = 'MNIST'
    num_clients = 20
    batch_size = 64
    max_data_size = 10000

    # Create the federated dataset
    fed_dataset = FedDataset(dataset_name, path_dataset, transform, max_data_size)

    # Define the trainloader
    trainloader = fed_dataset.create_clients_loader(num_clients, batch_size)

    # Define the testloader
    testloader = fed_dataset.testloader
