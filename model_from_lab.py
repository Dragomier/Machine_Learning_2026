import torch
import torchvision
import torch

transform = torchvision.transforms.Compose(
    [ torchvision.transforms.ToTensor(), #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
      torchvision.transforms.Normalize((0.1307), (0.3081))])

trainset = torchvision.datasets.MNIST(root='./data',
                                      train=True,
                                      download=True,
                                      transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=2048,
                                          shuffle=True)   #we do shuffle it to give more randomizations to training epochs

testset = torchvision.datasets.MNIST(root='./data',
                                     train=False,
                                     download=True,
                                     transform=transform)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=1,
                                         shuffle=False)

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()                   # we need only one ReLU instance, because ReLU has no parameters.
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(1*28*28, 1024) # this is the first Linear layer
        self.linear2 = torch.nn.Linear(1024, 2048)
        self.linear3 = torch.nn.Linear(2048, 256)
        self.linear4 = torch.nn.Linear(256, 10)

        self.dropout = torch.nn.Dropout(0.05)

    def forward(self, x):      # B, 1, 28, 28
        x = self.flatten(x)    # B, 784

        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu(x)

        x = self.linear3(x)
        x = self.relu(x)

        x = self.linear4(x)

        x = self.dropout(x)
        return x


def train_network():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Working on {device}")

    net = MLP().to(device)
    optimizer = torch.optim.Adam(net.parameters(), 0.001)  # initial and fixed learning rate of 0.001.

    net.train()  # it notifies the network layers (especially batchnorm or dropout layers, which we don't use in this example) that we are doing traning
    for epoch in range(8):  # an epoch is a training run through the whole data set

        for batch, data in enumerate(trainloader):
            batch_inputs, batch_labels = data

            batch_inputs = batch_inputs.to(device)  # explicitly moving the data to the target device
            batch_labels = batch_labels.to(device)

            # batch_inputs.squeeze(1)     #alternatively if not for a Flatten layer, squeeze() could be used to remove the second order of the tensor, the Channel, which is one-dimensional (this index can be equal to 0 only)

            optimizer.zero_grad()

            batch_outputs = net(
                batch_inputs)  # this line calls the forward(self, x) method of the MLP object. Please note, that the last layer of the MLP is linear
            # and MLP doesn't apply
            # the nonlinear activation after the last layer
            loss = torch.nn.functional.cross_entropy(batch_outputs, batch_labels,
                                                     reduction="mean")  # instead, nonlinear softmax is applied internally in THIS loss function
            loss.backward()  # this computes gradients as we have seen in previous workshops
            optimizer.step()  # but this line in fact updates our neural network.
    return net

