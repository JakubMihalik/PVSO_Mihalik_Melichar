import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.cluster import Birch

pcd = o3d.io.read_point_cloud("Pointclouds/output.ply")

plane_model, inliers = pcd.segment_plane(distance_threshold=0.07, ransac_n=3, num_iterations=30000)
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
# o3d.visualization.draw_geometries([outlier_cloud])

while True:
    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.04, ransac_n=3, num_iterations=30000)
    inlier_cloud = outlier_cloud.select_by_index(inliers)
    outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([outlier_cloud])

    if len(outlier_cloud.points) < 274000:
        break

pcd_without_planes = outlier_cloud
data_points = np.asarray(pcd_without_planes.points)
# Train Birch model on the data points
birch = Birch(threshold=0.6, branching_factor=5, n_clusters=None)
birch.fit(data_points)

# Predikcia zhlukov pre dátové body
labels = birch.predict(data_points)
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd_without_planes.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd_without_planes])
