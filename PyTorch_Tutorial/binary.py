# binary classification.
from numpy import vstack
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# dataset class.
class CSVDataset(torch.utils.data.Dataset):
  def __init__(self, path):
    df = pd.read_csv(path, header=None)
    self.X = df.values[:, :-1]
    self.Y = df.values[:, -1]
    # cvt to float.
    self.X = self.X.astype('float32')
    # fix labels.
    self.Y = LabelEncoder().fit_transform(self.Y)
    self.Y = self.Y.astype('float32')
    self.Y = self.Y.reshape((len(self.Y), 1))

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return [self.X[idx], self.Y[idx]]

  def get_splits(self, n_test=0.33):
    test_size = round(n_test * len(self.X))
    train_size = len(self.X) - test_size
    return torch.utils.data.random_split(self, [train_size, test_size])

# model
class MLP(torch.nn.Module):
  def __init__(self, n_inputs):
    super(MLP, self).__init__()
    # layer 1.
    self.hidden1 = torch.nn.Linear(n_inputs, 10)
    torch.nn.init.kaiming_uniform_(self.hidden1.weight,
                                   nonlinearity='relu')
    self.act1 = torch.nn.ReLU()
    # layer 2.
    self.hidden2 = torch.nn.Linear(10, 8)
    torch.nn.init.kaiming_uniform_(self.hidden2.weight,
                                   nonlinearity='relu')
    self.act2 = torch.nn.ReLU()
    # head layer.
    self.hidden3 = torch.nn.Linear(8, 1)
    torch.nn.init.xavier_uniform_(self.hidden3.weight)
    self.act3 = torch.nn.Sigmoid()

  def forward(self, X):
    X = self.hidden1(X)
    X = self.act1(X)
    X = self.hidden2(X)
    X = self.act2(X)
    X = self.hidden3(X)
    X = self.act3(X)
    return X

def prepare_data(path):
  # load
  dataset = CSVDataset(path)
  # split
  train, test = dataset.get_splits()
  # prepare
  train_dl = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
  test_dl = torch.utils.data.DataLoader(test, batch_size=1024, shuffle=False)
  return train_dl, test_dl

def train_model(train_dl, model):
  # optimizaiton
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  # epochs
  for epoch in range(100):
    for i, (inputs, targets) in enumerate(train_dl):
      # clear grads.
      optimizer.zero_grad()
      # perform model.
      yhat = model(inputs)
      # get loss.
      loss = criterion(yhat, targets)
      # bp.
      loss.backward()
      # update.
      optimizer.step()


def evaluate_model(test_dl, model):
  predictions, actuals = list(), list()
  for i, (inputs, targets) in enumerate(train_dl):
    # perform model.
    yhat = model(inputs)
    # cvt to numpy.
    yhat = yhat.detach().numpy()
    actual = targets.numpy()
    catual = actual.reshape((len(actual), 1))
    # round
    yhat = yhat.round()
    # ?!
    predictions.append(yhat)
    actuals.append(actual)
  predictions, actuals = vstack(predictions), vstack(actuals)
  # acc
  acc = accuracy_score(actuals, predictions)
  return acc

def predict(row, model):
  # cvt
  row = torch.Tensor([row])
  yhat = model(row)
  yhat = yhat.detach().numpy()
  return yhat

# prepare the data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(34)
# train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
# make a single prediction (expect class=1)
row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,
       -0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,
       0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,
       -0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,
       0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
yhat = predict(row, model)
print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))
