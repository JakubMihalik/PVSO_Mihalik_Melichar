import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.cluster import Birch

pcd = o3d.io.read_point_cloud("Pointclouds/bathroom.ply")
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
# pcd.paint_uniform_color([0.6, 0.6, 0.6])
# #o3d.visualization.draw_geometries([pcd]) #Works only outside Jupyter/Colab
# # plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,ransac_n=3,num_iterations=1000)
# plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,ransac_n=3,num_iterations=2000)
# [a, b, c, d] = plane_model
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
# inlier_cloud = pcd.select_by_index(inliers)
# outlier_cloud = pcd.select_by_index(inliers, invert=True)
# inlier_cloud.paint_uniform_color([1.0, 0, 0])
# outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
# ODSTRANENIE PLANOV
plane_model, inliers = pcd.segment_plane(distance_threshold=0.07, ransac_n=3, num_iterations=30000)
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([outlier_cloud])

while True:
    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.04, ransac_n=3, num_iterations=30000)
    inlier_cloud = outlier_cloud.select_by_index(inliers)
    outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
    o3d.visualization.draw_geometries([outlier_cloud])

    if len(outlier_cloud.points) < 274000:
        break

pcd_without_planes = outlier_cloud
data_points = np.asarray(pcd_without_planes.points)
# Train Birch model on the data points
birch = Birch(threshold=0.8, branching_factor=5, n_clusters=None)
birch.fit(data_points)

# Predikcia zhlukov pre dátové body
labels = birch.predict(data_points)
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd_without_planes.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd_without_planes])

# o3d.visualization.draw_geometries([pcd_without_planes])
# -------------------------- SEGMENT
# segment_models, inliers = pcd_without_planes.segment_plane(distance_threshold=0.04, ransac_n=3, num_iterations=3000)
# segments = pcd_without_planes.select_by_index(inliers)

# data_points = np.asarray(segments.points)
# # Inicializujte BIRCH algoritmus
# birch = Birch(threshold=5,branching_factor=50,n_clusters=None)

# # Natrénujte BIRCH model na dátových bodoch
# birch.fit(data_points)

# # Predikcia zhlukov pre dátové body
# labels = birch.predict(data_points)
# max_label = labels.max()
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# pcd_without_planes.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd_without_planes])

# labels = np.array(pcd.cluster_dbscan(eps=0.04, min_points=10))
# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")

# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
# segment_models={}
# segments={}
# max_plane_idx=1
# rest=pcd_without_planes
# d_threshold=0.01
# # BIRCH ALGORITMUS
# for i in range(max_plane_idx):
#     colors = plt.get_cmap("tab20")(i)
#     segment_models[i], inliers = rest.segment_plane(distance_threshold=0.04, ransac_n=3, num_iterations=5000)
#     segments[i] = rest.select_by_index(inliers)

#     # Konvertujte dáta na numpy array pre použitie BIRCH algoritmu
#     data_points = np.asarray(segments[i].points)
    
#     # Inicializujte BIRCH algoritmus
#     birch = Birch(threshold=1,branching_factor=50,n_clusters=None)
    
#     # Natrénujte BIRCH model na dátových bodoch
#     birch.fit(data_points)
    
#     # Predikcia zhlukov pre dátové body
#     labels = birch.predict(data_points)

#     candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]
#     best_candidate = int(np.unique(labels)[np.argmax(candidates)])
#     print("the best candidate is: ", best_candidate)
#     rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(list(np.where(labels != best_candidate)[0]))
#     segments[i] = segments[i].select_by_index(list(np.where(labels == best_candidate)[0]))
#     segments[i].paint_uniform_color(list(colors[:3]))
#     print("pass", i+1, "/", max_plane_idx, "done.")
    
# o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])
