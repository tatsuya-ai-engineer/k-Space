import numpy as np
import faiss
import torch
from sklearn import metrics

class FaissKNeighbors:
    def __init__(self, k):
        self.index = None
        self.y = None
        self.k = k
        
        # print(faiss.StandardGpuResources())

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        # このvotesの中に獲得票数が入っている。
        votes = self.y[indices]
        self.votes = votes
        
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions
    
    def get_each_prediction_and_votes(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]

        pred_array = np.empty((X.shape[0], 0))
        for one_k in range(self.k):
            pred = np.array([np.argmax(np.bincount(x[:one_k+1])) for x in votes])
            pred_array = np.append(pred_array, pred.reshape(-1,1), axis=1)
        
        # indicesは次のような結果を返す
        # inc[0][k] => k+1番目に近い訓練データの、訓練データのインデックスを返す
        return indices, pred_array, votes
    
    def get_votes(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        
        return votes