#!/usr/bin/env python
'''
This script applies hierarchical clustering implementation with complete linkage to the data.
Author: Sai Venkata Krishnaveni Devarakonda
Date: 04/29/2022
'''
import numpy as np
from utilities import Load_data,hierarchical_clustering,most_common_label
str_path_1b_program = './data_2c2d3c3d_program.txt'

features, targets = Load_data(str_path_1b_program)
c = [2, 4, 6, 8]
for c in c:
    clusters, unique_val = hierarchical_clustering(features, "complete", c)

    clusters_with_labels = np.column_stack((clusters, targets))
    targets_clusters = np.zeros([c, len(clusters_with_labels)], dtype='str')

    for i in range(len(unique_val)):
        for j in range(clusters_with_labels.shape[0]):
            if (clusters[j] == unique_val[i]):
                targets_clusters[i][j] = clusters_with_labels[j][1]

    accuracy = 0
    for i in range(c):
        temp = targets_clusters[i]
        index_str = np.argwhere(temp == '')
        temp = np.delete(temp, index_str)
        l, counts = np.unique(temp, return_counts=True)
        wt_frac = len(temp) / len(targets)
        # print(wt_frac)

        # find most frequent label
        label = most_common_label(temp)

        counter = 0
        for j in range(len(temp)):
            if (label == temp[j]):
                counter = counter + 1
        wt_frac_2 = counter / len(temp)
        # print(wt_frac_2)

        accuracy = accuracy + (wt_frac * wt_frac_2)
        # print(accuracy)
    accuracy = accuracy * 100

    print('Accuracy for hierarchical clustering of complete linkage  with ' + str(c) + ' clusters is ' + str(accuracy))

