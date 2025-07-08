import os
import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull, distance_matrix

def read_swc(filepath):
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            x, y, z = map(float, parts[2:5])  
            points.append([x, y, z])
    return np.array(points)

def plot_convex_hulls(pre_coords, post_coords):
    plotter = pv.Plotter()
    
    hull_pre = ConvexHull(pre_coords)
    hull_post = ConvexHull(post_coords)
    
    pre_hull_mesh = pv.PolyData(pre_coords[hull_pre.vertices]).delaunay_3d()
    post_hull_mesh = pv.PolyData(post_coords[hull_post.vertices]).delaunay_3d()
    
    try:
        intersection = pre_hull_mesh.boolean_intersection(post_hull_mesh)
        has_intersection = intersection.n_points > 0
    except Exception as e:
        print(f"Intersection computation failed: {e}")
        has_intersection = False
    
    plotter.add_mesh(pre_hull_mesh, color="red", opacity=0.5, label="Pre Neurites")
    plotter.add_mesh(post_hull_mesh, color="blue", opacity=0.5, label="Post Neurites")
    
    if has_intersection:
        plotter.add_mesh(intersection, color="green", opacity=0.7, label="Intersection")
        print(f"Intersection Volume: {intersection.volume}")
    else:
        print("No intersection detected.")
    
    plotter.add_legend()
    plotter.show()

def analyze_relative_position(pre_coords, post_coords):
    centroid_pre = np.mean(pre_coords, axis=0)
    centroid_post = np.mean(post_coords, axis=0)
    distance = np.linalg.norm(centroid_pre - centroid_post)
    
    print(f"Centroid of Pre Neurites: {centroid_pre}")
    print(f"Centroid of Post Neurites: {centroid_post}")
    print(f"Distance between centroids: {distance}")

    distances = distance_matrix(pre_coords, post_coords)
    min_distance = np.min(distances)
    print(f"Minimum distance between pre and post neurites: {min_distance}")

pre_neurites_path = r"H:\energency_fly\data\pre_test"
post_neurites_path = r"H:\energency_fly\data\post_test"

for filename in os.listdir(pre_neurites_path):
    pre_file = os.path.join(pre_neurites_path, filename)
    post_file = os.path.join(post_neurites_path, filename)
    
    if os.path.exists(post_file):
        print(f"Processing {filename}...")

        pre_coords = read_swc(pre_file)
        post_coords = read_swc(post_file)

        if len(pre_coords) < 4 or len(post_coords) < 4:
            print(f"Skipping {filename}: Not enough points for ConvexHull.")
            continue

        analyze_relative_position(pre_coords, post_coords)

        plot_convex_hulls(pre_coords, post_coords)
