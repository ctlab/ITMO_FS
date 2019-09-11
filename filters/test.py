# from math import log
#
# import numpy as np
# from scipy import sparse as sp
# from sklearn.metrics import mutual_info_score as MI_features
#
#
# def powerset(s):
#     x = len(s)
#     masks = [1 << i for i in range(x)]
#     for i in range(1 << x):
#         yield [ss for mask, ss in zip(masks, s) if i & mask]
#
#
# def contingency_matrix(labels_true, labels_pred):
#     classes, class_idx = np.unique(labels_true, return_inverse=True)
#     clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
#     n_classes = classes.shape[0]
#     n_clusters = clusters.shape[0]
#     # Using coo_matrix to accelerate simple histogram calculation,
#     # i.e. bins are consecutive integers
#     # Currently, coo_matrix is faster than histogram2d for simple cases
#     contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
#                                  (class_idx, cluster_idx)),
#                                 shape=(n_classes, n_clusters),
#                                 dtype=np.int)
#     contingency = contingency.tocsr()
#     contingency.sum_duplicates()
#     return contingency
#
#
# def __mi(U, V):
#     contingency = contingency_matrix(U, V)
#     nzx, nzy, nz_val = sp.find(contingency)
#     contingency_sum = contingency.sum()
#     pi = np.ravel(contingency.sum(axis=1))
#     pj = np.ravel(contingency.sum(axis=0))
#     log_contingency_nm = np.log(nz_val)
#     contingency_nm = nz_val / contingency_sum
#     # Don't need to calculate the full outer product, just for non-zeroes
#     outer = (pi.take(nzx).astype(np.int64, copy=False)
#              * pj.take(nzy).astype(np.int64, copy=False))
#     log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
#     mi = (contingency_nm * (log_contingency_nm - log(contingency_sum)) +
#           contingency_nm * log_outer)
#     return mi.sum()
#
#
# def t(a, b):
#     print(a, end="\n")
#     print(b)
#     print(MI_features(a, b))
#     print(__mi(a, b))
#     print("\n")
#
#
# # t([1, 2], [1, 2])
# # t([1, 1, 2], [1, 1, 2])
# # t([1, 1, 1, 2], [1, 1, 1, 2])
# # t([1, 1, 1, 1, 2], [1, 1, 1, 1, 2])
# # t([1, 2, 2], [1, 2, 2])
# # t([1, 2, 3], [1, 2, 3])
# # t([1, 2, 3], [3, 1, 2])
# t([1, 2, 3, 1], [3, 1, 2, 1])
# t([1, 2, 1, 3], [3, 1, 1, 2])
# t([1, 2, 4, 3],
#   [3, 1, 1, 5])
# # t([1], [1])
# # t([2, 2], [2, 2])
