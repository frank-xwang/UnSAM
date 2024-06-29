import torch.nn.functional as F
import numpy as np
import torch

def merged_clusters(i, j, clusters):
    c1, c2 = clusters[i], clusters[j]
    weighted_sum = ((c1['feature'] + c2['feature']) / (c1['num_of_patch'] + c2['num_of_patch'])).float()
    #weighted_sum = ((c1['num_of_patch']*c1['feature'] + c2['num_of_patch']*c2['feature']) / (c1['num_of_patch'] + c2['num_of_patch'])).float()
    return {
        'feature': c1['feature'] + c2['feature'],
        'normalized_feature': F.normalize(weighted_sum, dim=0),
        'mask': (c1['mask'] + c2['mask']) > 0,
        'num_of_patch': c1['num_of_patch'] + c2['num_of_patch'],
        'neighbors': c1['neighbors'].union(c2['neighbors']).difference(set([i, j]))
    }

def iterative_merge(features, threshes, min_size=4):

    clusters = []
    similarities = {}
    H, W = features.shape[:2]

    cluster_idx = 0
    for y in range(H):
        for x in range(W):
            mask = np.zeros((H, W))
            mask[y, x] = 1
            clusters.append({
                'feature': features[y, x],
                'normalized_feature': F.normalize(features[y, x].float(), dim=0),
                'mask': mask,
                'num_of_patch': 1,
                'neighbors': set()
            })

            if (cluster_idx % W) != 0:
                clusters[cluster_idx]['neighbors'].add(cluster_idx-1)
                clusters[cluster_idx-1]['neighbors'].add(cluster_idx)
                similarities[(cluster_idx-1, cluster_idx)] = \
                    torch.dot(clusters[cluster_idx-1]['normalized_feature'], clusters[cluster_idx]['normalized_feature']).item()
            if (cluster_idx - W) >= 0:
                clusters[cluster_idx]['neighbors'].add(cluster_idx-W)
                clusters[cluster_idx-W]['neighbors'].add(cluster_idx)
                similarities[(cluster_idx-W, cluster_idx)] = \
                    torch.dot(clusters[cluster_idx-W]['normalized_feature'], clusters[cluster_idx]['normalized_feature']).item()
                
            cluster_idx += 1

    all_masks = []     
    for thresh in threshes:
        while len(similarities):
            i, j = max(similarities, key=similarities.get)
            if similarities[(i, j)] < thresh: break
            
            merged = merged_clusters(i, j, clusters)
            clusters.append(merged)

            del similarities[(i, j)]
            for neighbor in merged['neighbors']:
                if i in clusters[neighbor]['neighbors']:
                    if neighbor < i: del similarities[(neighbor, i)]
                    else: del similarities[(i, neighbor)]
                    clusters[neighbor]['neighbors'].discard(i)
                if j in clusters[neighbor]['neighbors']:
                    if neighbor < j: del similarities[(neighbor, j)]
                    else: del similarities[(j, neighbor)]
                    clusters[neighbor]['neighbors'].discard(j)

                similarities[(neighbor, cluster_idx)] = \
                    torch.dot(clusters[neighbor]['normalized_feature'], clusters[cluster_idx]['normalized_feature']).item()
                clusters[neighbor]['neighbors'].add(cluster_idx)

            cluster_idx += 1
        
        single_level_masks = []
        counted_cluster = set()
        for (m, n) in similarities:
            if m not in counted_cluster:
                counted_cluster.add(m)
                single_level_masks.append(clusters[m]['mask']) if clusters[m]['num_of_patch'] >= min_size else None
            if n not in counted_cluster:
                counted_cluster.add(n)
                single_level_masks.append(clusters[n]['mask']) if clusters[n]['num_of_patch'] >= min_size else None
        all_masks.append(np.stack(single_level_masks)) if len(single_level_masks) else None
        
    return all_masks