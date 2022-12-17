"""
EE 298 Machine Problem 1
REMOVING PROJECTIVE DISTORTION ON IMAGES
Instructions:
1.) Download requirements.txt
2.) Run solver.py

"""

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r" ,package])

from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
import csv
import matplotlib.pyplot as plt
import random

def data_reader(filename):
  data = []
  with open(filename,'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    for row in csv_reader:
      if row[0] == 'x': continue
      data.append(row)
    csv_file.close()
  return data

def data_splitter(data, percent_split):
  random.shuffle(data) 
  split1 = data[:round(percent_split*len(data))]
  split2 = data[round(percent_split*len(data)):]
  return split1, split2

class PolynomialSolver():
  def __init__(self,n, epoch, batch_size, batch_number):
    self.n = n
    self.epoch = epoch
    self.batch_size = batch_size
    self.coeff = Tensor.uniform(self.n+1,1)
    self.batch_number = batch_number

  def forward(self,x):
    y_calc = 0
    for i in range(self.n + 1):
      y_calc = y_calc + self.coeff[i]*(x**i)
    return y_calc

  def train(self,data,optim):
    counter = 0
    for i in range(self.epoch):
      optim.zero_grad()
      if counter == max(data.shape): counter = 0
      loss = 0
      for j in range(self.batch_number):
        prediction = self.forward(data[counter][0])
        iter_loss = (data[counter][1] - prediction)**2 #Mean Square Error
        loss = loss + iter_loss
        counter = counter + 1
      loss = (1/self.batch_number)*loss
      loss.backward()
      optim.step()  
      print(loss.data)

  def eval(self,data):
    loss = 0
    counter = 0
    for i in range(max(data.shape)):
      prediction = self.forward(data[counter][0])
      iter_loss = (data[counter][1] - prediction)**2 #Mean Square Error
      loss = iter_loss + loss
      counter = counter + 1
    loss = (1/max(data.shape)) * loss
    print(loss.grad)



data_train = data_reader('data_train.csv')
data_test = data_reader('data_test.csv')
data_train, data_validate = data_splitter(data_train, 0.9)

n = 1   # Degree Number
epoch = 20 # Number of Epoch
batch_size = 2 # Number of Data per Batch
batch_number = round(len(data_train)/batch_size)

data_train = Tensor(data_train, requires_grad = True)
data_validate = Tensor(data_validate, requires_grad = True)

model = PolynomialSolver(n, epoch, batch_size, batch_number)
optim = optim.SGD([model.coeff],lr = 0.01)
model.train(data_train,optim)

