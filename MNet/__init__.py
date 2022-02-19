from time import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from MNet.net import MNet

# neural net parameters
input_size = 784 # 28*28
hidden_size = 100
output_size = 10
num_epochs = 100
batch_size = 100
learning_rate = 0.01

def run():
  print(f'\nLearning Rate: {learning_rate}\nEpochs: {num_epochs}\nBatch Size: {batch_size}\n')

  # device configuration
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # data transformer
  transform = transforms.Compose([transforms.ToTensor()])

  # train data
  train_data = torchvision.datasets.MNIST(root='./MNet/data', 
                                          train=True, 
                                          transform=transform, 
                                          download=True)
                                          
  test_data = torchvision.datasets.MNIST(root='./MNet/data', 
                                          train=False, 
                                          transform=transform)

  # data loaders
  train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                            batch_size=batch_size, 
                                            shuffle=True)

  test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                            batch_size=batch_size, 
                                            shuffle=False)

  # examples = iter(train_loader)
  # samples, labels = examples.next()
  # print(samples.shape, labels.shape)

  # model
  model = MNet(input_size, hidden_size, output_size)

  # loss + optimizer
  loss_fn = nn.CrossEntropyLoss()
  optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # train
  n_total_steps = len(train_loader)
  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
      images = images.reshape(-1, 28*28).to(device)
      labels = labels.to(device)

      # forward
      out = model(images)
      loss = loss_fn(out, labels)

      # backward
      optim.zero_grad()
      loss.backward()
      optim.step()

      if (i+1) % 100 == 0:
        print(f'Epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss {loss.item():.4f}')

  # test
  with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader:
      images = images.reshape(-1, 28*28).to(device)
      labels = labels.to(device)
      out = model(images)

      _, predictions = torch.max(out, 1)
      n_samples += labels.shape[0]
      n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy {acc}')

  model_num = int(time())
  torch.save(model.state_dict(), f'./models/mnist_api_{model_num}.pth')
  print(f'\nNew Model: {model_num}\n')