# import numpy as np

# def distance(a, b):
#     return np.sqrt(np.sum((a-b) ** 2))

# def get_neighbours(data, point_indx, eps):
#     neighbours = []
#     for i in range(len(data)):
#         if distance(data[point_indx], data[i] < eps):
#             neighbours.append(i)
#     return neighbours
        
# def dbscan(data, eps, min_samples):
#     labels = [-1] * len(data)
#     cluster_id = 0
    
#     for i in range(len(data)):
#         if labels[i] != -1:
#             continue
        
#         neighbours = get_neighbours(data, i, eps)
#         if len(neighbours) < min_samples:
#             labels[i] = -1
#             continue
        
#         labels[i] = cluster_id
#         search_queue = [n for n in neighbours if n != i]
#         while search_queue:
#             current_point = search_queue.pop(0)
#             if labels[current_point] == -1:
#                 labels[current_point] = cluster_id
#             elif labels[current_point] != cluster_id:
#                 continue
#             new_neighbours = get_neighbours(data, current_point, eps)
#             new_neighbours = [n for n in new_neighbours if n != current_point]

#             if len(new_neighbours) >= min_samples:
#                 for n in new_neighbours:
#                     if n not in search_queue:
#                         search_queue.append(n)
#         cluster_id += 1
    
#     return labels
            
    
# # Example usage
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs

# data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

# plt.scatter(data[:, 0], data[:, 1])
# plt.show()

# # labels = dbscan(data, eps=0.5, min_samples=10)

# # plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
# # plt.show()

import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def get_neighbors(data, point_idx, eps):
    neighbors = []
    for i, data_point in enumerate(data):
        if euclidean_distance(data[point_idx], data_point) < eps:
            neighbors.append(i)
    return neighbors

def dbscan(data, eps, min_samples):
    labels = [-1] * len(data)
    cluster_id = 0

    for point_idx in range(len(data)):
        if labels[point_idx] != -1:
            continue

        neighbors = get_neighbors(data, point_idx, eps)

        if len(neighbors) < min_samples:
            labels[point_idx] = -1
            continue

        labels[point_idx] = cluster_id
        search_queue = [n for n in neighbors if n != point_idx]

        i = 0
        while i < len(search_queue):
            current_point = search_queue[i]

            if labels[current_point] == -1:
                labels[current_point] = cluster_id

            if labels[current_point] == cluster_id:
                new_neighbors = get_neighbors(data, current_point, eps)
                new_neighbors = [n for n in new_neighbors if n != current_point]

                if len(new_neighbors) >= min_samples:
                    for n in new_neighbors:
                        if n not in search_queue:
                            search_queue.append(n)

            i += 1

        cluster_id += 1

    return labels

# Example usage
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

data, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.6, random_state=0)
labels = dbscan(data, eps=0.7, min_samples=10)

plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.show()