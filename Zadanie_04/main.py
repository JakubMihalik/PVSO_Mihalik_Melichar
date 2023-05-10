import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.cluster import Birch

pcd = o3d.io.read_point_cloud("Pointclouds/bathroom.ply")
# o3d.visualization.draw_geometries([pcd])


# Apply the color map to the point cloud
# plane_model, inliers = pcd.segment_plane(distance_threshold=0.06, ransac_n=3, num_iterations=1000)
# inlier_cloud = pcd.select_by_index(inliers)
# outlier_cloud = pcd.select_by_index(inliers, invert=True)
# inlier_cloud.paint_uniform_color([1, 0, 0])
# outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
# # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
# print("[Clustering]")
# labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))
# max_label = labels.max()
# colors = plt.get_cmap("tab20")(labels / (max_label 
# if max_label > 0 else 1))
# colors[labels < 0] = 0
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=10000)
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([outlier_cloud])

while True:
    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.04, ransac_n=3, num_iterations=3000)
    inlier_cloud = outlier_cloud.select_by_index(inliers)
    outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)

    if len(outlier_cloud.points) < 230000:
        break

pcd_without_planes = outlier_cloud
o3d.visualization.draw_geometries([pcd_without_planes])

# DBSCAN
labels = np.array(pcd_without_planes.cluster_dbscan(eps=0.1, min_points=10))
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd_without_planes.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd_without_planes])

# segment_models={}
# segments={}
# max_plane_idx=40
# rest=pcd
# d_threshold=0.001
# for i in range(max_plane_idx):
#     if len(rest.points) < 3:
#         break
#     colors = plt.get_cmap("tab20")(i)
#     plane_model, inliers = rest.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=3000)
#     segments[i] = rest.select_by_index(inliers)
#     labels = np.array(segments[i].cluster_dbscan(eps=d_threshold*10, min_points=30))
#     candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]
#     best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0]][0])
#     print("the best candidate is: ", best_candidate)
#     rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(list(np.where(labels != best_candidate)[0]))
#     segments[i] = segments[i].select_by_index(list(np.where(labels == best_candidate)[0]))
#     segments[i].paint_uniform_color(list(colors[:3]))
#     print("pass", i+1, "/", max_plane_idx, "done.")
# labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")

# colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# rest.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([segments[i] for i in segments.keys()]+[rest])

# d_threshold = 5
# for i in range(max_plane_idx):
#     labels = np.array(segments[i].cluster_dbscan(eps=d_threshold*10, min_points=10))
#     candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)]
#     candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)]
#     best_candidate=int(np.unique(labels)[np.where(candidates== np.max(candidates))[0]])
#     rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(list(np.where(labels!=best_candidate)[0]))
#     segments[i]=segments[i].select_by_index(list(np.where(labels== best_candidate)[0]))
#     labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
#     max_label = labels.max()
#     print(f"point cloud has {max_label + 1} clusters")
#     colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
#     colors[labels < 0] = 0
#     rest.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
# o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])

# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
# pcd.paint_uniform_color([0.6, 0.6, 0.6])
# # o3d.visualization.draw_geometries([pcd])

# plane_model, inliers = pcd.segment_plane(distance_threshold=0.06,ransac_n=3,num_iterations=1000)
# [a, b, c, d] = plane_model
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
# inlier_cloud = pcd.select_by_index(inliers)
# outlier_cloud = pcd.select_by_index(inliers, invert=True)
# inlier_cloud.paint_uniform_color([1.0, 0, 0])
# outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))
# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")

# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# o3d.visualization.draw_geometries([pcd])

# segment_models={}
# segments={}
# max_plane_idx=10

# rest=pcd
# for i in range(max_plane_idx):
#     colors = plt.get_cmap("tab20")(i)
#     plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,ransac_n=3,num_iterations=1000)
#     segments[i]=rest.select_by_index(inliers)
#     segments[i].paint_uniform_color(list(colors[:3]))
#     rest = rest.select_by_index(inliers, invert=True)
#     print("pass",i,"/",max_plane_idx,"done.")

# o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])
# segment_models={}
# segments={}
# max_plane_idx=20
# rest=pcd
# d_threshold=0.01
# for i in range(max_plane_idx):
#     colors = plt.get_cmap("tab20")(i)
#     plane_model, inliers = rest.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)
#     segments[i] = rest.select_by_index(inliers)
#     labels = np.array(segments[i].cluster_dbscan(eps=d_threshold*10, min_points=10))
#     candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]
#     best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0]])
#     print("the best candidate is: ", best_candidate)
#     rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(list(np.where(labels != best_candidate)[0]))
#     segments[i] = segments[i].select_by_index(list(np.where(labels == best_candidate)[0]))
#     segments[i].paint_uniform_color(list(colors[:3]))
#     print("pass", i+1, "/", max_plane_idx, "done.")
    
# labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")

# colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

# o3d.visualization.draw_geometries([segments.values()])
# o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])
# o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])
# o3d.visualization.draw_geometries([rest])


