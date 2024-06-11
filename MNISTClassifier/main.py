import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    batch_size = 100

    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    test_x = Variable(torch.unsqueeze(test_dataset.data, dim=1)).type(torch.FloatTensor)
    test_y = test_dataset.targets

    print(train_dataset)
    print('\n')
    print(test_dataset)
    print('-----')

    print(train_loader)
    print(test_loader)

    class Net(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    # Neurons of each layer

    input_size = 784
    hidden_size = 500
    num_classes = 10

    # Initialization
    MLP = Net(input_size, hidden_size, num_classes)

    # Hyperparameters
    num_epochs = 5
    learning_rate = 0.001

    # Loss function and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(MLP.parameters(), lr=learning_rate)


    def correctness(predictions, labels):
     
        pred = torch.max(predictions.data, 1)[1]
        correct = pred.eq(labels.data.view_as(pred)).sum()
        return correct, len(labels)

    record = []  

    for epoch in range(num_epochs):

        train_correct = []  

        for i, (images, labels) in enumerate(train_loader):

            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            MLP.train()

            optimizer.zero_grad()  
            outputs = MLP(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            correct = correctness(outputs, labels)  
            train_correct.append(correct)

            if (i + 1) % 200 == 0:
                MLP.eval() 

                train_c = (sum([tup[0] for tup in train_correct]), sum([tup[1] for tup in train_correct]))
                train_accuracy = 100. * train_c[0].numpy() / train_c[1]
                total_step = len(train_dataset) // batch_size

                print('Epoch [{:d}/{:d}], Step [{:3d}/{:d}], Loss: {:.4f} | training accuracy: {:5.2f} %'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.data, train_accuracy))

                record.append(100 - 100. * train_c[0] / train_c[1])

    # Evaluate Test Set
    correct = 0
    total = 0
    actual_labels = []
    predicted_labels = []
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28 * 28))
        outputs = MLP(images)
        _, predicted = torch.max(outputs.data, 1)
        actual_labels.extend(labels.numpy())
        predicted_labels.extend(predicted.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print(classification_report(actual_labels, predicted_labels))
    
    isExist = os.path.exists("./ConfusionMatrices")
    if not isExist:
       os.makedirs("./ConfusionMatrices")

    confusion = confusion_matrix(actual_labels, predicted_labels)
    ConfusionMatrixDisplay(confusion).plot()
    plt.savefig('./ConfusionMatrices/confusion_matrix.png')
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
