import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.cluster import Birch

pcd = o3d.io.read_point_cloud("Pointclouds/output.ply")

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
labels = np.array(pcd_without_planes.cluster_dbscan(eps=0.2, min_points=10))
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd_without_planes.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd_without_planes])