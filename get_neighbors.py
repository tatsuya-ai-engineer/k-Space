import pickle 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics

import torch

from fast_kNN import FaissKNeighbors

#################################################################

model_name = "resnet"
train_name = "IN"
test_name = "IN"


classes = ['airplane', 'bear', 'bicycle', 'bird', 
          'boat', 'bottle', 'car', 'cat', 
          'chair', 'clock', 'dog', 'elephant',
          'keyboard', 'knife', 'oven', 'truck']


# load train
train_path = f'softmax_dict/{train_name}_to_{model_name}-{train_name}_softmax_dict.pkl'
with open(train_path, mode='rb') as f:
    dicts = pickle.load(f)

train_X = np.empty((0, 1000))
train_y = np.empty((0, ))

for i, c in enumerate(classes):
  train_X = np.append(train_X, dicts[c], axis=0).astype(float)
  train_y = np.append(train_y, [i]*dicts[c].shape[0], axis=0).astype(int)

print("traininig k-model")
k = train_X.shape[0]
knn_model = FaissKNeighbors(k=k)
knn_model.fit(train_X, train_y)


# load test
test_path = f'softmax_dict/{test_name}_to_{model_name}-{train_name}_softmax_dict.pkl'
with open(test_path, mode='rb') as f:
    dicts = pickle.load(f)

votes_dict = {}
pred_dict = {}

for c_num, c in enumerate(classes):
  test_X = dicts[c]

  # get neighbors
  print(f" ===> Get {c} k votes")

  v_all = np.empty((0, train_X.shape[0]))
  p_all = np.empty((0, train_X.shape[0]))
  for i, one_test_X in enumerate(test_X):
      pred, votes = knn_model.get_each_prediction_and_votes(one_test_X[np.newaxis, :])
      v_all = np.append(v_all, votes, axis=0)
      p_all = np.append(p_all, pred, axis=0)
    
  votes_dict[c] = v_all
  pred_dict[c] = p_all

save_path = f'neighbor_dict/{test_name}_to_{model_name}-{train_name}_neighbors_dict.pkl'
with open(save_path, 'wb') as p:
    pickle.dump(votes_dict , p)

save_path = f'prediction_dict/{test_name}_to_{model_name}-{train_name}_prediction_dict.pkl'
with open(save_path, 'wb') as p:
    pickle.dump(pred_dict , p)