import math
import time
from typing import Dict, List, Tuple

import numpy as np

from src.features.custom_clustering import DistanceMatrixUtils, DistanceMatrix


def split_cluster(cluster, dist):
    """
    Splits a single custer into 2 using the diana algorithm.
    :param cluster: a dict with keys nodes -> [list],
    :param dist:
    :return:
    """
    total_nodes = len(cluster['nodes'])
    base_nodes = total_nodes

    base_cluster = cluster['nodes']  # this becomes the first cluster (cluster A)
    new_cluster = []  # this becomes the second cluster (cluster B)

    sum_base_dists = cluster['sum_dists']  # sum of distance for each node to current cluster (= base cluster)
    # sum of distance for each node to new cluster (0, because new cluster is empty so distance to itself/None)
    sum_new_dists = [0] * total_nodes

    if sum_base_dists is None:
        # not pre-calculated, so manual calculation of current distances
        sum_base_dists = [
            sum([dist[node_from][node_to] for node_to in base_cluster]) for node_from in base_cluster
        ]

    # now we try to find a cluster to splinter
    while base_nodes > 1:
        new_nodes = total_nodes - base_nodes
        max_node_id = -1
        max_node_delta = -1
        if base_nodes == total_nodes:
            # first iteration, so just pick the worst node to make its own cluster
            delta_distance = sum_base_dists
            max_node_id = np.argmax(delta_distance)
            # noinspection PyTypeChecker
            max_node_delta = delta_distance[max_node_id]
        else:
            # iterate over all the nodes in the base cluster (by index)
            for id in range(len(sum_base_dists)):
                sumT = sum_base_dists[id]  # get distance to the base cluster
                sumB = sum_new_dists[id]  # get distance to the new cluster
                delta = (sumT - sumB) * new_nodes - sumB * (base_nodes - 1)  # calculate if moving improves the distance
                # from sum(A)/len(A) - sum(B)/len(B)
                # = sum(A) * len(B) - sum(B) * len(A)
                # = (sum(T) - sum(B)) * len(B) - sum(B) * len(A)
                # with len(A) = base_nodes - 1
                if delta > max_node_delta:
                    max_node_delta = delta
                    max_node_id = id

        if max_node_delta <= 0 and base_nodes != total_nodes:
            # Splitting does not improve the clusters, so stop splitting
            # second part is so that clusters with 2 nodes where the distances are 0 are still split
            break

        # Append the newly found node to the new cluster
        new_cluster.append(base_cluster[max_node_id])

        # Update the sum of distances to the new cluster
        dist_row = dist[base_cluster[max_node_id]]
        sum_new_dists = [sum_new_dists[i] + dist_row[base_cluster[i]]
                         for i in range(base_nodes) if not i == max_node_id]
        # and remove the node from the old cluster
        del sum_base_dists[max_node_id]
        del base_cluster[max_node_id]
        base_nodes = len(base_cluster)
        # note: we do not change the sum_base_dists because these are the total distances, not the inside A distances
        # doing it this way prevents a recalculation of these distances on every iteration

    sum_base_dists = [sumT - sumB for (sumT, sumB) in zip(sum_base_dists, sum_new_dists)]
    return base_cluster, new_cluster, sum_base_dists


# Find the index of the evaluation result where the average result (in the surrounding window of size `window_size`)
# is the largest. This helps when the evaluation metric is not 'smooth'.
def get_optimal_num_clusters(eval_results, window_size=5):
    # if we do not have enough observations to calculate averages, decrease window size or return first value
    if len(eval_results) == 0:
        print("[WARNING] no evaluation results! Assuming 2 clusters")
        return 2
    while len(eval_results) <= 0.5 * window_size:
        window_size -= 1
        print("[WARNING] window size decreased with 1 to %d" % window_size)

    # Found an appropriate window size -> calculate
    max_evaluation_value = min(eval_results)  # set to lowest value
    max_evaluation_index = 0
    for i in range(len(eval_results)):
        # Get the evaluation values surrounding our current index
        eval_values = [value for index, value in enumerate(eval_results)
                       if index - 0.5 * window_size <= i <= index + 0.5 * window_size]
        if len(eval_values) == 0:
            # No items
            continue
        # calculate average and compare to current best
        average_value = sum(eval_values) / len(eval_values)
        if max_evaluation_value < average_value:
            max_evaluation_value = average_value
            max_evaluation_index = i

    # our eval_results starts at index 0 for 2 clusters, then index 1 for 3 clusters, etc
    # so to determine the number of clusters we need to add 2
    return max_evaluation_index + 2


def get_diameter(cluster, dist):
    assert dist is not None
    dia = -1
    for i in cluster:
        for j in cluster:
            if dist[i][j] > dia:
                dia = dist[i][j]
    return dia


# Pre-calculates values used for modularity score calculation
# TODO (maybe) verify we do not have overflow/rounding problems
# The original paper has integer distances (0 <= dist <= 100), so they sidestep a lot of these issues
# at the cost of less accurate values (and it's probably a lot quicker: integer maths >> float maths)
def precompute_modularity_values(dist):
    total_sum = 0
    sum_rows = []
    for row in dist:
        sum_row = sum([1 - row[i] for i in range(len(dist))]) - 1
        # to prevent counting similarity/distance to self, subtract (1 - 0) = 1
        sum_rows.append(sum_row)
        total_sum += sum_row
    return total_sum, sum_rows


# Calculates the modularity of a clustering
# TODO verify correctness (especially: are all edges counted the correct number of times)
# I've been staring at this, but I don't understand how it works.
# Need eyes on this I guess. For now, we just assume the original code did it correctly
def calc_modularity(clusters, dist, precomputed_modularity_values):
    total_degree = precomputed_modularity_values[0]
    if total_degree == 0:
        return -0.5

    degree = precomputed_modularity_values[1]
    first_entry = 0  # coverage
    second_entry = 0

    # For details on the definition, read 'On modularity clustering', Brandes et al., 2016
    for cluster in clusters:
        nodes = cluster["nodes"]
        for node_a in nodes:
            dist_row = dist[node_a]
            first_entry -= 1 - dist_row[node_a]  # remove weights of edges to self
            for node_b in nodes:
                first_entry += 1 - dist_row[node_b]  # sum all edge weights inside the cluster twice
                second_entry += degree[node_a] * degree[node_b]  # (sum deg i)^2 = sum_i sum_j deg i * deg j

    return (first_entry - second_entry / total_degree) / total_degree


# Function to calculate the difference in modularity (quicker than full recalculation)
# TODO verify correctness
# see `calc_modularity`
def calc_modularity_shift(cluster_a, cluster_b, dist, precomputed_modularity_values):
    total_sum = precomputed_modularity_values[0]
    if total_sum == 0:
        return -0.5
    sum_rows = precomputed_modularity_values[1]
    # num_nodes = len(dist)
    first_entry = 0
    second_entry = 0
    for node_a in cluster_a:
        dist_row = dist[node_a]
        for node_b in cluster_b:
            first_entry += 1 - dist_row[node_b]
            second_entry += sum_rows[node_a] * sum_rows[node_b]
    return -2 * (first_entry - second_entry / total_sum) / total_sum


def get_idf_map(freqs: List[Dict], index_to_user_list, verify_not_normalized=True):
    if verify_not_normalized:
        for user, freq in freqs.items():
            for act, count in freq.items():
                if 0 < count < 1:
                    # if a value is a count, it is either 0, or larger than 1
                    # if it is normalized, it has 0 < val < 1
                    print("[WARNING] idf_map calculation needs activity counts, not normalized frequencies!")
    idf_map = {}
    num_users = len(index_to_user_list)
    for user in index_to_user_list:
        total_activity_count = sum(freqs[user].values())  # these values must NOT be normalized!
        for activity, frequency in freqs[user].items():
            # update value if exists, otherwise set to 0 and then update
            # TODO: why is this weighted
            idf_map[activity] = idf_map.setdefault(activity, 0) + frequency / total_activity_count
    for activity, frequency in idf_map.items():
        # weigh by log inverse frequency
        idf_map[activity] = math.log(num_users / frequency)
    return idf_map


def exclude_features(feature_weights, exclusion=None):
    """
    Excludes a set of features by setting their weight to 0
    :param feature_weights: a map containing feature importance weights
    :param exclusion: a list containing features to be excluded
    :return: the feature_weights map with excluded features mapped to weight 0
    """
    if exclusion is not None:
        for feature in exclusion:
            feature_weights[feature] = 0
    return feature_weights


def get_halfpoint(feature_scores: List[Tuple[int, float]]):
    """
    Returns the point at which 3/4ths of the total score is covered.
    :param feature_scores: an ordered (descending by score) list containing the tuples (index, score)
    :return: the index of the feature that splits the score in 75%-25%
    """
    score_only = [abs(score) for index, score in feature_scores]
    total_score = sum(score_only) * 3 / 4
    return np.argmax(np.cumsum(score_only) > total_score)


def create_tf_idf(clusters: List[Dict], freqs: Dict, idf_map: Dict, index_to_user_list: List[int], current_exclusions):
    # Prevent iteration over None
    if current_exclusions is None:
        current_exclusions = ()

    full_tf_idf = {}
    cluster_sizes = {}
    sum_tf_idf = {}
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        # transforms the nodes in the cluster to indices in the frequency list
        user_list = sorted([index_to_user_list[index] for index in cluster["nodes"]])

        # Calculate the tf_idf of the activities in this cluster
        # Note, the idf of the activities are pre-calculated w.r.t. the root cluster
        # excluded features are ignored
        tf_idf = {}
        for user in user_list:  # user (= document) contains activities (= words)
            for activity in freqs[user].keys():
                if activity in current_exclusions:
                    continue
                idf = idf_map[activity]  # calculated w.r.t. root cluster (higher = more unique)
                count = freqs[user][activity]  # raw count; not frequency
                tf_idf[activity] = tf_idf.setdefault(activity, []) + [
                    count * 1]  # should be able to just have 1 value
                sum_tf_idf[activity] = sum_tf_idf.setdefault(activity, 0) + count * 1
                # note: tf_idf[activity] contains a list of tf_idf scores, one for each user/node in the cluster
                # TODO: is this stable/can we link users to their frequencies with this or do we need a dict
                # also, do we even need to link users to their frequencies??

        for activity in current_exclusions:
            if activity in tf_idf:
                print(f"[WARNING] Excluded activity ({activity}) in tf_idf map for cluster {cluster_id}!")

        cluster_sizes[cluster_id] = len(user_list)
        full_tf_idf[cluster_id] = tf_idf
    return cluster_sizes, full_tf_idf, sum_tf_idf


def get_chi2_feature_scores(clusters, freqs, idf_map, index_to_user_list, current_exclusions,
                            print_feature_score=False):
    """
    For each feature not excluded, calculates its tf.idf score (idf w.r.t. the root cluster). This score is then used
    to determine which features are most important in a cluster (e.g. if some feature has a high tf in some cluster,
    but a low tf in another cluster, that feature is important to the first cluster). The selection mechanism is based
    on the chi2 score of the feature. This function returns an ordered (descending) list of these features and their
    score for each cluster.
    :param clusters: list of clusters as created by the diana algorithm
    :param freqs: complete dictionary of activity frequencies for each user
    :param idf_map: map of inverse document frequencies of each activity
    :param index_to_user_list: list to convert cluster nodes to root cluster indices
    :param current_exclusions: ignored features
    :param print_feature_score: whether to print the scores calculated
    :return: a dict containing for each cluster_id in cluster all features and their chi2 score in descending order
    """

    # Calculate feature counts (as tf.idf, instead of simple counts)
    cluster_sizes, combined_tf_idf, sum_tf_idf = create_tf_idf(
        clusters, freqs, idf_map, index_to_user_list, current_exclusions)

    # set up some basic variables
    total_nodes = sum(cluster_sizes.values())
    score_map = {}
    for cluster_id, tf_idf in combined_tf_idf.items():
        scores = {}  # contains feature -> score mapping; will be sorted on scores later
        for activity, scores_list in tf_idf.items():  # here, activity = feature
            # we now have a chi2 goodness of fit test, where we see if the feature is distributed differently for
            # the clusters. The idea is to figure out if some feature is independent of classification or not. If it
            # is not independent of the classification, it was probably important in the classificaiton
            # that means we remove it, so that for the next step in hierarchical clustering we look only at the features
            # that have not been used.
            # Note, we should in fact bin the continuous variables but we do not do that here
            # https://stats.stackexchange.com/questions/24179/how-exactly-does-chi-square-feature-selection-work
            tot = sum_tf_idf[activity]
            inside = sum(scores_list)
            outside = tot - inside

            expected_in = cluster_sizes[cluster_id] / total_nodes * tot
            expected_out = tot - expected_in

            # TODO combine bins if expected size < 5 (is this actually necessary? It just means it's a great score)
            # for now just warn
            if expected_out < 1 or expected_in < 1:
                pass
                # print(f"[WARNING] Found bin with expected count < 1 in cluster {cluster_id}, activity {activity}")
            elif expected_out < 5 or expected_in < 5:
                pass
                # print(f"[WARNING] Found bin with expected count < 5 in cluster {cluster_id}, activity {activity}")

            if expected_in == 0 or expected_out == 0:
                # this should not happen
                print(f"[WARNING] expected in or out is 0: in {expected_in} out {expected_out}")
                chi2 = float('inf')
            else:
                # otherwise calculate straightforward score
                chi2 = (inside - expected_in) ** 2 / expected_in + (outside - expected_out) ** 2 / expected_out

            # To see whether a score results from an absence or over
            if inside < expected_in:
                scores[activity] = -chi2
            else:
                scores[activity] = chi2

        # convert dict to ordered list of tuples
        score_map[cluster_id] = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)

        if print_feature_score:
            for feature, score in score_map[cluster_id]:
                print(f"cluster: {cluster_id} feature: {feature} score: {score}")

    return score_map


def get_exclusion_map(clusters, freqs, idf_map, index_to_user_list, current_exclusions):
    # determine feature scoring for each feature in each created cluster
    score_map = get_chi2_feature_scores(clusters, freqs, idf_map, index_to_user_list,
                                        current_exclusions)

    exclusion_map = {}
    exclusion_score_map = {}
    for cluster_id, feature_scores in score_map.items():
        # print(feature_scores)
        # for each cluster, we determine what the important features are by finding the 'elbow' the scores
        # i.e. the point after which each feature contributes only a little bit to the overall clustering
        index_score = [(i, abs(x[1])) for i, x, in enumerate(feature_scores)]
        # first, we find the number of features to consider for the LMethod
        # the paper described that taking all features into account leads to a worse result, as the elbow
        # is usually found earlier
        halfpoint = get_halfpoint(feature_scores)  # set of features that together make up 75+% of the score

        # plt.plot([feature_scores[i][1] for i in range(len(feature_scores))])
        # plt.show()
        # pick the largest of halfpoint, 200 but no more than the number of features we have
        knee = get_sweetspot_L_method(index_score[:max(2 * halfpoint, 200)])

        if knee == 2 and len(feature_scores) > 2:
            # second score is more than 3x smaller than first
            #and abs(feature_scores[1][1]) * 3 < abs(feature_scores[0][1])
            delta1 = abs(feature_scores[0][1]) - abs(feature_scores[1][1])
            delta2 = abs(feature_scores[1][1]) - abs(feature_scores[2][1])
            if delta2 * 5 < delta1:
                # big difference between increases in knee
                knee -= 1

        # we always remove 2+ features with the L method. Is it important to be able to remove less?
        exclusion_map[cluster_id] = [x[0] for i, x in enumerate(feature_scores) if i < knee]
        exclusion_score_map[cluster_id] = [x[1] for i, x in enumerate(feature_scores) if i < knee]

    return exclusion_map, exclusion_score_map


def get_residuals(x, y):
    """
    Retrieves the residuals of linear regression (with intercept) of x and y
    :param x: x (1d)
    :param y: y
    :return: sum of residuals
    """
    if len(x) <= 2:
        return 0

    a = np.vstack([x, np.ones(len(x))]).T
    residuals = np.linalg.lstsq(a, y, rcond=None)[1]
    return residuals[0] if len(residuals) > 0 else 0


# Calculates the elbow/knee in the modularity using the L-method
# Paper: https://sci-hub.st/10.1109/ictai.2004.50 (Determining the Number of Clusters/Segments in Hierarchical
# Clustering/Segmentation Algorithms, Salvador and Chan)
def get_knee_L_method(x, y):
    if len(x) < 4:
        # need 4 points to create 2 lines
        return len(x)
        # return int(len(x) / 2 + 0.5)

    if len(x) != len(y):
        print("[WARNING] len x != len y in L method!")
        return len(x)

    lowest_residual = np.inf
    knee = None
    # the knee to test is the leftmost feature of the right section
    # so it is part of the part before the knee (i.e. if the knee is found to be after x = 2, knee = 3)
    for knee_to_test in range(2, len(x) - 1):
        # for knee_to_test in range(1, len(x)):
        # we solve the linear regression with matrices
        left_residuals = get_residuals(np.array(x[:knee_to_test]), np.array(y[:knee_to_test]))
        right_residuals = get_residuals(np.array(x[knee_to_test:]), np.array(y[knee_to_test:]))

        if right_residuals + left_residuals < lowest_residual:
            lowest_residual = right_residuals + left_residuals
            knee = knee_to_test
    return knee


def get_sweetspot_L_method(index_score_tuples):
    """
    Iteratively changes the window of the L-method to arrive at a better estimate for the knee. The L-method performs
    best when there are approx. as many points left and right of the knee.
    :param index_score_tuples: a list of tuples (index, feature, score), ordered by score (descending) (and index, asc)
    :return: the index of the feature that defines the knee
    """
    if len(index_score_tuples) < 1:
        return 0

    x = [index for index, score in index_score_tuples]
    y = [score for index, score in index_score_tuples]

    cutoff = len(index_score_tuples)
    current_knee = cutoff
    last_knee = current_knee + 1
    while current_knee < last_knee:
        last_knee = current_knee
        current_knee = get_knee_L_method(x[:cutoff], y[:cutoff])
        cutoff = 2 * current_knee

    return current_knee


def _get_min_cluster_size(total_nodes, size_threshold, min_cluster_size):
    """Calculates the absolute minimum cluster size, based on relative minimum, absolute minimum and useful minimum (2,
    since we cannot split smaller clusters)"""
    return max(total_nodes * size_threshold, min_cluster_size, 2)


def run_divisive_clustering(freqs, size_threshold=0.05, min_cluster_size=0, initial_dist=None, num_clusters=None, idf_map=None,
                            exclusions=None, index_to_user_list=None, max_threads=4):
    """

    :param min_cluster_size: minimum cluster size (stop clustering when all clusters are smaller than this)
    :param freqs: list of activity frequencies per user (MUST NOT BE NORMALIZED)
    :param size_threshold: minimum size (rel) after which clusters are no longer split
    :param initial_dist: pre-computed distance matrix. Will be calculated if not provided.
    :param num_clusters: clusters to create in each step, if None the number is determined automatically (modularity)
    :param idf_map: a map containing the idfs (inverse document frequencies) of the activities of the users in the root cluster.
        You can think of an user as a document and his/her activities as a word. Note: if idf_map is None, the map will
        be calculated and will take the exclusions into account.
    :param exclusions: features excluded from the distance calculation
    :param index_to_user_list: a mapping of indices to actual users, when considering nested clusters (e.g. a nested cluster
        may contain only users [2,7,18] and will simply rename these to [0,1,2]. With this map, we can go back to [2,7,18].
        The map is in this example is the list [2,7,18], defining that index 0, 1, 2 correspond to user 2, 7, 18 resp.
    :return:
    """

    # handle no exclusions
    if exclusions is None:
        exclusions = []

    # create default index_to_user mapping
    if index_to_user_list is None:
        index_to_user_list = [i for i in range(len(freqs))]

    # calculate default feature weights
    if idf_map is None:
        idf_map = exclude_features(
            get_idf_map(freqs, index_to_user_list, verify_not_normalized=True),
            exclusions)

    # set or create distance matrix
    if initial_dist is not None:
        dist = initial_dist
    else:
        freqs_normalized = {user: DistanceMatrixUtils.normalize_freq(freq) for user, freq in freqs.items()}
        # dist = DistanceMatrixUtils.calc_distance_matrix_multithreaded(freqs_normalized, num_threads=2)
        dist = DistanceMatrix.calc_distance_matrix(freqs_normalized, num_threads=max_threads)

    # setup initial cluster
    total_nodes = len(dist)
    cluster_id = 0
    child_parent_map = []
    clusters = [
        {"nodes": [i for i in range(total_nodes)],
         "diameter": 1,  # does not matter for first iteration, select max
         "sum_dists": None,
         "cluster_id": cluster_id}
    ]
    # precompute values used in clustering evaluation
    precomputed_modularity_values = precompute_modularity_values(dist)
    evaluation_results = []

    # Split the cluster in half until all clusters have 0 diameter or the largest cluster is smaller than the minimum
    # size
    # note that `clusters` only contains the current clusters. We merge them back after selecting the optimal
    # number of clusters based on the modularity
    # TODO: investigate other stopping criteria (e.g. decrease in modularity?)
    action_time = time.time()
    print(f"[INFO] Splitting clusters as deep as possible...")
    while (clusters[-1]["diameter"] > 0  # cluster does not contain only identical users and size is not too small
           and len(clusters[-1]["nodes"]) >= _get_min_cluster_size(total_nodes, size_threshold, min_cluster_size)):
        parent_cluster_id = clusters[-1]["cluster_id"]
        child_parent_map.append((parent_cluster_id, cluster_id + 1, cluster_id + 2))

        # clustering the last (biggest) cluster into 2 new clusters (and also returns the total distance inside a)
        cluster_a, cluster_b, cluster_a_sum_dist = split_cluster(clusters.pop(), dist)

        # store the new clusters
        cluster_id += 1
        clusters.append({"nodes": cluster_a, "diameter": get_diameter(cluster_a, dist),
                         "sum_dists": cluster_a_sum_dist, "cluster_id": cluster_id})
        cluster_id += 1
        # note that we do not have the inter-cluster distance for cluster b; we will calculate it once we need it
        clusters.append({"nodes": cluster_b, "diameter": get_diameter(cluster_b, dist),
                         "sum_dists": None, "cluster_id": cluster_id})

        # sort clusters by diameter, then size (if the cluster is smaller than the minimum size, set the diameter to 0)
        def sorting(x):
            if len(x['nodes']) > _get_min_cluster_size(total_nodes, size_threshold, min_cluster_size):
                return (x['diameter'], len(x['nodes']))
            else:
                return (0, 0)

        clusters = sorted(clusters, key=sorting)
        # clusters = sorted(clusters,
        #                   key=lambda x: (x['diameter'], len(x['nodes'])) if len(x['nodes']) >= _get_min_cluster_size(total_nodes, size_threshold, min_cluster_size) else (0, 0),
        #                   )

        # Evaluate clustering performance
        if len(evaluation_results) == 0:
            # first evaluation -> full calculation
            eval_result = calc_modularity(clusters, dist, precomputed_modularity_values)
        else:
            eval_result = evaluation_results[-1] + calc_modularity_shift(cluster_a, cluster_b, dist,
                                                                         precomputed_modularity_values)
        evaluation_results.append(eval_result)

        # print(clusters)
    print(f"[INFO] Clusters split. Time spent: {time.time() - action_time:.2f}s.")
    # Clustering is now complete. To determine the number of clusters, we figure out which configuration has
    # the optimal (= highest) modularity (or we take the parameter from the call, if given)
    if num_clusters is None:
        num_clusters = get_optimal_num_clusters(evaluation_results)

    # print("Determined optimal number of clusters (%d), with modularity %.4f"
    #       % (num_clusters, evaluation_results[num_clusters - 2]))

    # Merge clusters, so we are left with the number of clusters as determined by the sweetspot
    print(f"[INFO] Merging clusters based on optimal number of splits...")
    action_time = time.time()
    cluster_map = {cluster["cluster_id"]: cluster for cluster in clusters}  # map clusters by id
    while len(clusters) > num_clusters:
        # retrieve most recent split
        parent_cluster_id, child_a_id, child_b_id = child_parent_map.pop()
        # merge children into parent
        parent_nodes = cluster_map[child_a_id]["nodes"] + cluster_map[child_b_id]["nodes"]
        parent_cluster = {"nodes": parent_nodes,
                          "diameter": 0,
                          "sum_dists": None,
                          "cluster_id": parent_cluster_id}
        # update mapping and list of clusters
        cluster_map[parent_cluster_id] = parent_cluster
        clusters.append(parent_cluster)
        clusters.remove(cluster_map[child_a_id])
        clusters.remove(cluster_map[child_b_id])
    # print("Finished merging clusters. Current situation: ")
    # print(clusters)

    print(f"[INFO] Done merging clusters. Time spent {time.time() - action_time:.2f}s.")
    print(f"[INFO] Determining feature score for each cluster...")
    action_time = time.time()
    # Now, we determine the defining features of each cluster and remove them.
    cluster_ids = [cluster['cluster_id'] for cluster in clusters]
    exclusion_map, exclusion_scores_map = get_exclusion_map(clusters, freqs, idf_map, index_to_user_list, exclusions)
    print(f"[INFO] Done determining feature scores. Time spent {time.time() - action_time:.2f}s.")
    results = []
    # For each cluster, we now set up some stuff and continue clustering
    # TODO: can we maybe use multiprocessing for this?
    for cluster_index in range(len(clusters)):
        cluster = clusters[cluster_index]
        child_users = cluster['nodes']  # contains indices of parent cluster, not users themselves
        child_index_to_user_list = sorted(
            [index_to_user_list[node] for node in child_users])  # create list of index -> user
        child_exclusions = exclusion_map[cluster['cluster_id']]
        child_exclusion_scores = exclusion_scores_map[cluster['cluster_id']]

        result = ('l', child_index_to_user_list, {"exclusions": child_exclusions,
                                                  "exclusion_scores": child_exclusion_scores})
        # Check if we need to continue clustering this child cluster
        if len(child_users) >= _get_min_cluster_size(total_nodes, size_threshold, min_cluster_size):
            all_child_exclusions = child_exclusions + exclusions  # exclusions = child + parent exclusions
            child_exclusion_set = set(all_child_exclusions)  # no doubles
            ignored_users = [user for user in child_index_to_user_list
                             # ignore users that have only excluded activities
                             if len(set(freqs[user].keys()) - child_exclusion_set) == 0]
            child_index_to_user_list = sorted(
                list(set(child_index_to_user_list) - set(ignored_users)))  # all users - ignored users
            if len(child_index_to_user_list) >= _get_min_cluster_size(total_nodes, size_threshold, min_cluster_size):
                # cluster is large enough to continue
                # TODO: this is single-threaded. Rework to make multi-threaded?
                # throw away frequencies that we don't use anymore
                minimal_freqs = {user: freq for user, freq in freqs.items() if user in child_index_to_user_list}
                # new_dist = DistanceMatrixUtils.calc_partial_distance_matrix(minimal_freqs,
                #                                                             indices_from=child_index_to_user_list,
                #                                                             exclude_features=all_child_exclusions,
                #                                                             full_width=False)
                new_dist = DistanceMatrix.calc_distance_matrix(minimal_freqs,
                                                               indices_from=child_index_to_user_list,
                                                               indices_to=child_index_to_user_list,
                                                               exclude_features=all_child_exclusions,
                                                               num_threads=max_threads)
                if np.max(new_dist) > 0:
                    # if the feature removal yields perfect similar users, don't continue clustering them!
                    result = run_divisive_clustering(minimal_freqs,
                                                     initial_dist=new_dist,
                                                     exclusions=all_child_exclusions,
                                                     index_to_user_list=child_index_to_user_list,
                                                     min_cluster_size=min_cluster_size)

                    info = result[2]
                    # if len(result) > 2:
                    #     info = result[2]
                    # else:
                    #     info = {}

                    # these are the users that have no features left to cluster
                    if len(ignored_users) > 0:
                        result[1].append(('l', ignored_users, {'isExclude': True}))

                    info['exclusions'] = child_exclusions
                    info['exclusion_scores'] = child_exclusion_scores
                    result = (result[0], result[1], info)

        results.append(result)
    return 't', results, {'sweetspot': evaluation_results[num_clusters - 2]}


# tree
# type = (l)eaf or sub-(t)ree
#   subtree if t
#   nodes if l
# extra info = {}
def result_to_tree(tree, start_cid=1, activity_id_to_name_map=None, node_to_user_id=None, _current_path=None):
    """Converts the output from the recursive clustering procedure into a nicely structured tree,

    activity_id_to_name_map is a dict containing id -> name mappings of activities
    If ngrams is false, we assume that each exclusion is precisely 1 activity.
    If ngrams is true, each exclusion is a tuple containing multiple activities"""
    if _current_path is None:
        _current_path = []
    # parse leaf
    if tree[0] == 'l':
        if len(tree[1]) == 0:
            return start_cid, None
        nodes = tree[1]
        if node_to_user_id is not None:
            nodes = [node_to_user_id[node] for node in nodes]
        parsed_exclusions, parsed_exclusion_scores = parse_exclusions(tree[2], activity_id_to_name_map)
        result = {'type': 'l',
                  'nodes': nodes,
                  'size': len(nodes),
                  'exclusions': parsed_exclusions,
                  'exclusions_scores': parsed_exclusion_scores,
                  'path': ".".join(_current_path + [str(start_cid)]),
                  'id': str(start_cid)}
        return start_cid + 1, result

    # parse (sub)trees
    sub_trees = []
    cid = start_cid
    start_cid += 1
    for i, sub_tree in enumerate(tree[1]):
        # _current_path.append(str(i))
        start_cid, new_sub_tree = result_to_tree(sub_tree, start_cid, activity_id_to_name_map, node_to_user_id, _current_path + [str(cid)])
        # _current_path.pop()
        if new_sub_tree is not None:
            sub_trees.append(new_sub_tree)

    parsed_exclusions, parsed_exclusion_scores = parse_exclusions(tree[2], activity_id_to_name_map)
    result = {'type': 't',
              'children': sub_trees,
              'exclusions': parsed_exclusions,
              'exclusions_scores': parsed_exclusion_scores,
              'path': ".".join(_current_path + [str(cid)]),
              'id': str(cid)}
    return start_cid, result


def parse_exclusions(tree_2, activity_id_to_name_map=None):
    """Parses the exclusions (from ids to names) and their scores"""
    if 'exclusions' not in tree_2:
        # No exclusions -> is ignored user group
        parsed_exclusions = "No exclusions (probably top-level or ignored users)!"
        parsed_exclusion_scores = []
    else:
        ngrams = isinstance(tree_2['exclusions'][0], tuple)
        parsed_exclusion_scores = tree_2['exclusion_scores']
        if activity_id_to_name_map is None:
            # No mapping; so just use numbers directly
            parsed_exclusions = tree_2['exclusions']
        elif not ngrams:
            # Not tuples, so translate each exclusion directly
            parsed_exclusions = [activity_id_to_name_map[act] for act in tree_2['exclusions']]
        else:
            # ngrams -> translate each element of each tuple in the exclusion list
            parsed_exclusions = [str(tuple(activity_id_to_name_map[act] for act in act_tuple))
                                 for act_tuple in tree_2['exclusions']]

    return parsed_exclusions, parsed_exclusion_scores
