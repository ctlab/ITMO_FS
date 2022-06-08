import numpy as np

from frozenlist import FrozenList
from scipy.spatial import Delaunay
from sortedcontainers import SortedSet
from functools import partial, cmp_to_key
from sklearn.model_selection import cross_val_score


class ExactMelif:
    _K_FOLD = 5
    _RANDOM_STATE = 42

    def __init__(self, measures, kappa, estimator, score):
        self._measures = measures
        self._kappa = kappa
        self._estimator = estimator
        self._score = score

    def fit(self, X, y):
        planes = self._get_planes(X, y)
        planes = self._normalize(planes)
        planes = self._kappa_filter(planes)

        intersections = self._get_intersections(planes)
        edges = self._get_edges(planes, intersections)

        self._best_feature_indices = self._get_best_feature_indices(edges, X, y)
        self._estimator.fit(self._best_sub_X(X), y)

    def predict(self, X):
        return self._estimator.predict(self._best_sub_X(X))

    def _get_planes(self, X, y):
        n_objects, n_features = X.shape
        n_measures = len(self._measures)
        planes = np.empty((n_features, n_measures))

        for j in range(n_measures):
            score = self._measures[j](X, y)
            for i in range(n_features):
                planes[i][j] = score[i]

        return planes

    @staticmethod
    def _normalize(planes):
        shape = planes.shape
        normalized = np.zeros(shape)

        minimum = planes.min()
        maximum = planes.max()
        min_max_diff = maximum - minimum

        for i in range(shape[0]):
            for j in range(shape[1]):
                normalized[i][j] = (planes[i][j] - minimum) / min_max_diff

        return normalized

    def _kappa_filter(self, planes):
        n_measures = len(self._measures)

        indexed = []
        for i, plane in enumerate(planes):
            indexed.append((i, plane))
        planes = np.array(indexed, dtype=object)

        kappa_indices = set()
        for i in range(n_measures):
            planes = sorted(planes, key=lambda p: p[1][i])
            kappa_indices.add(planes[-self._kappa][0])

        filtered_indices = set()
        for i in range(n_measures):
            planes.sort(key=lambda p: p[1][i])

            left = 0
            while planes[left][0] not in kappa_indices:
                left += 1

            right = len(planes) - 1
            while planes[right][0] not in kappa_indices:
                right -= 1

            for j in range(left, right + 1):
                filtered_indices.add(planes[j][0])

        filtered_planes = []
        for i, plane in planes:
            if i in filtered_indices:
                filtered_planes.append(plane)

        return np.array(filtered_planes)

    def _get_intersections(self, planes):
        n_planes, dim = planes.shape
        intersections = np.zeros((n_planes, n_planes), dtype=np.ndarray)

        for i in range(n_planes):
            for j in range(i + 1, n_planes):
                plane_i = planes[i]
                plane_j = planes[j]

                intersection = SortedSet(key=cmp_to_key(_double_list_cmp))
                for k in range(dim):
                    for l in range(k + 1, dim):
                        a, b = plane_i[k], plane_i[l]
                        c, d = plane_j[k], plane_j[l]

                        if abs(a * d - b * c) < 1e-9:
                            continue

                        point = np.zeros(dim)
                        point[k] = a * c * (d - b) / (a * d - b * c)
                        point[l] = b * d * (c - a) / (b * c - a * d)

                        if point[k] < 0 or point[k] > 1 \
                                or point[l] < 0 or point[l] > 1:
                            continue

                        point = FrozenList(point)
                        point.freeze()
                        intersection.add(point)

                intersection = list(intersection)
                for k in range(len(intersection)):
                    intersection[k] = list(intersection[k])

                intersections[i][j] = intersection
                intersections[j][i] = intersection

        return intersections

    def _get_edges(self, planes, intersections):
        n_planes, dim = planes.shape
        edges = []

        for i in range(n_planes):
            plane_points = []

            for j in range(dim):
                point = np.zeros(dim)
                point[j] = planes[i][j]
                plane_points.append(point)

            for j in range(n_planes):
                if i != j:
                    for point in intersections[i][j]:
                        if point != 0:
                            plane_points.append(point)

            plane_points = np.array(plane_points)
            if len(plane_points) == 3:
                edges.append(plane_points)
            else:
                triangulation = Delaunay(plane_points, qhull_options='QJ Pp')
                edges.extend(plane_points[triangulation.simplices])

        return edges

    def _get_best_feature_indices(self, edges, X, y):
        best_feature_indices = None
        best_quality = None
        feature_indices_sets = set()

        for edge in edges:
            master_measure = self._build_master_measure(edge)
            feature_indices, filtered_X = self._filter_X(master_measure, X, y)

            if feature_indices in feature_indices_sets:
                continue

            feature_indices_sets.add(feature_indices)
            quality = self._get_quality(filtered_X, y)

            if best_quality is None or best_quality < quality:
                best_feature_indices = feature_indices
                best_quality = quality

        return best_feature_indices

    def _build_master_measure(self, edge):
        n_points, dim = edge.shape
        center = np.zeros(dim)
        for point in edge:
            for i in range(dim):
                center[i] += point[i]

        for i in range(dim):
            center[i] /= n_points

        alphas = center

        def master(measures, X, y):
            n_measures = len(self._measures)
            n_features = X.shape[1]
            result = np.zeros(n_features)

            for i in range(n_measures):
                value = measures[i](X, y)
                for j in range(n_features):
                    result[j] += alphas[i] * value[j]

            return result

        return partial(master, self._measures)

    def _filter_X(self, master_measure, X, y):
        features = np.transpose(X)
        n_features = len(features)

        scores = master_measure(X, y)
        feature_scores = []
        for feature_i, score in enumerate(scores):
            feature_scores.append((feature_i, score))

        feature_scores.sort(key=lambda p: p[1])

        feature_indices = set()
        for i in range(n_features - self._kappa, n_features):
            feature_indices.add(feature_scores[i][0])

        filtered_features = []
        for i, feature in enumerate(features):
            if i in feature_indices:
                filtered_features.append(feature)

        feature_indices = frozenset(feature_indices)
        filtered_features = np.array(filtered_features)
        filtered_X = np.transpose(filtered_features)

        return feature_indices, filtered_X

    def _get_quality(self, X, y):
        return np.mean(cross_val_score(self._estimator, X, y, scoring=self._score, cv=self._K_FOLD))

    @staticmethod
    def _sub_X(feature_indices, X):
        features = np.transpose(X)
        filtered_features = []

        for i, feature in enumerate(features):
            if i in feature_indices:
                filtered_features.append(feature)

        filtered_features = np.array(filtered_features)
        return np.transpose(filtered_features)

    def _best_sub_X(self, X):
        return self._sub_X(self._best_feature_indices, X)


def _double_list_cmp(a_list, b_list):
    for i in range(len(a_list)):
        res = _double_cmp(a_list[i], b_list[i])
        if res != 0:
            return res
    return 0


def _double_cmp(a, b):
    if abs(a - b) < 1e-9:
        return 0
    if a > b:
        return 1
    return -1
