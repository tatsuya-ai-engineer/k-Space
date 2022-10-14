import pickle 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn import metrics

import torch

from k_utils.ks import FaissKNeighbors

#################################################################

target_dataset = ['IN', 'SIN']
model_name = ['resnet-IN', 'resnet-SIN', 'vit-IN']

classes = ['airplane', 'bear', 'bicycle', 'bird', 
              'boat', 'bottle', 'car', 'cat', 
              'chair', 'clock', 'dog', 'elephant',
              'keyboard', 'knife', 'oven', 'truck']

proc_name = ['rotate', 'crop', 'noise']


# load train
for pn in proc_name:
    for dn in target_dataset:
        
        dis_dict_list = []
        fig = plt.figure(figsize=(16,16))
        for mn in model_name:
            # train
            train_dir = '/home/ueda-tatsuya/デスクトップ/k-Spectrum/model-vs-human/softmax_dict_1013/'
            with open(train_dir + f'originIN_to_{mn}.pkl', mode='rb') as f:
                train_list = pickle.load(f)

            train_X = train_list[0]
            train_y = train_list[1].astype("int64")

            print("traininig k-model")
            k = train_X.shape[0]
            knn_model = FaissKNeighbors(k=k)
            knn_model.fit(train_X, train_y)

            
            # test
            test_dir = '/home/ueda-tatsuya/デスクトップ/k-Spectrum/model-vs-human/softmax_dict_1013/'
            with open(test_dir + f'{pn}{dn}_to_{mn}.pkl', mode='rb') as f:
                test_dict = pickle.load(f)



            votes_dict = {}

            for c_num, c in enumerate(classes):
                test_X = test_dict[c]

                # get neighbors
                v_list = []
                v_all = np.empty((0, train_X.shape[0]))
                for i, one_test_X in enumerate(test_X):
                    votes = knn_model.get_votes(one_test_X)
                    
                    v_list.append(votes)
                    
                votes_dict[c] = v_list

                # plt.plot(range(i+1), origin_idx_list, marker="o")
                # plt.show()
            save_path = f'/home/ueda-tatsuya/デスクトップ/k-Spectrum/model-vs-human/softmax_dict_1013/neighbor_dict/{pn}{dn}_to_{mn}_neighbors_dict.pkl'
            with open(save_path, 'wb') as p:
                pickle.dump(votes_dict , p)
    
    
    
    
    
    
    
    
    
#             distance_dict = {}
#             for ci, c in enumerate(classes):
#                 test_X_list = test_dict[c]
#                 test_X = test_X_list[1]
#                 if pn=='crop':
#                     test_X = test_X[::-1]
#                 origin_idx = test_X_list[0]
#                 print(origin_idx, c)
                
#                 proc_distances = np.empty((0,))

                
#                 for i, one_test_X in enumerate(test_X):
#                     inc, pred, votes = knn_model.get_each_prediction_and_votes(one_test_X[np.newaxis, :])

#                     proc_distances = np.append(proc_distances, 
#                                                np.where(inc[0]==origin_idx))

                
#                 distance_dict[c] = proc_distances
                
#             dis_dict_list.append(distance_dict)
                
#         # create fig
#         for ci, c in enumerate(classes):
#             ax = fig.add_subplot(4,4,ci+1)
#             ax.plot(range(test_X.shape[0]), dis_dict_list[0][c], marker=".", label=model_name[0])
#             ax.plot(range(test_X.shape[0]), dis_dict_list[1][c], marker=".", label=model_name[1])
#             ax.plot(range(test_X.shape[0]), dis_dict_list[2][c], marker=".", label=model_name[2])
#             ax.set_ylim([-1, k])
#             ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#             ax.set_title(f"{c}")
#             ax.grid(axis='both')
#             ax.legend()
            
#         fig.suptitle(f'{pn}{dn}')
#         plt.tight_layout()
#         fig.savefig('/home/ueda-tatsuya/デスクトップ/k-Spectrum/model-vs-human/softmax_dict_1006/figs/' + \
#             f'{pn}{dn}')
#         # quit()




# save_path = f'prediction_dict/{test_name}_to_{model_name}-{train_name}_prediction_dict.pkl'
# with open(save_path, 'wb') as p:
#     pickle.dump(pred_dict , p)