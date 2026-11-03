import numpy as np

def k_nearest_neighbors(points, query_point, k):
    points = np.array(points)
    query_point = np.array(query_point)
    distances = np.linalg.norm(points - query_point, axis=1)
    indices = np.argsort(distances, kind='stable')
    return [tuple(points[i]) for i in indices[:k]]
