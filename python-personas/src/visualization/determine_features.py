import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import config
from config import timed

logger = logging.getLogger(__name__)


def _get_file(file=config.DATA_FOLDER_PROCESSED / 'no_acc_near_purchase_clusters.parquet'):
    return pd.read_parquet(file)


def _get_clusters(cluster):
    clusters = cluster.split(".")
    for i in range(len(clusters)):
        clusters = ".".join(clusters[:i + 1])
    return clusters


@timed
def get_feature_scores(df: pd.DataFrame, cluster_cols=None, feature_cols=None, max_depth=4):
    """Calculates chi_sq feature scores."""
    # setup cluster cols if missing
    if cluster_cols is None:
        cluster_cols = sorted(df.columns[np.where(df.columns.str.startswith("cluster"))].to_list())

    # setup feature cols if missing
    if feature_cols is None:
        feature_cols = []
        for col in df.columns:
            if col not in cluster_cols:
                feature_cols.append(col)

    # analyse features
    scores = {}
    for i, cluster_col in enumerate(cluster_cols):
        if i > max_depth:
            continue
        for cluster in df[cluster_col].unique():
            if cluster is None:
                continue
            logger.info(f"Analyzing features in cluster {cluster}")

            scores[cluster] = {}
            cluster_rows = (df[cluster_col] == cluster)
            if i == 0:
                # top level cluster
                parent_cluster_rows = pd.Series([True] * len(df), index=df.index)
            else:
                parent = ".".join(cluster.split(".")[:-1])
                parent_cluster_rows = df[cluster_cols[i - 1]] == parent

            # analyze one cluster in a specific cluster column
            for feature_col in feature_cols:
                scores[cluster][feature_col] = {}
                if is_numeric_dtype(df[feature_col].dtype):
                    # numeric variable
                    score = _calc_score(df, cluster_rows, parent_cluster_rows, feature_col)
                    if score is None:
                        continue
                    scores[cluster][feature_col][feature_col] = score
                else:
                    # not numeric, so look at each categorical var
                    features = df.loc[parent_cluster_rows, feature_col].unique()
                    if len(features) <= 1:
                        logger.debug(
                            f"Skipping feature scoring of column '{feature_col}' in cluster '{cluster}'. Only 1 feature!")
                        continue

                    # analyze one feature column in one cluster
                    skip = 0
                    for feature in features:
                        # analyze one specific feature in one cluster
                        result = _calc_score(df, cluster_rows, parent_cluster_rows, feature_col, feature)
                        if result is None:
                            skip += 1
                            continue
                        else:
                            scores[cluster][feature_col][feature] = result
                    logger.debug(f"Skipped analyzing {skip} features with E[in] < 5")
    return scores


def _calc_score(df, cluster_rows, parent_cluster_rows, feature_col, feature=None):
    if feature is not None:
        inside = (df.loc[cluster_rows, feature_col] == feature).sum()
        parent_cluster = (df.loc[parent_cluster_rows, feature_col] == feature).sum()
        all_with_feature = (df.loc[:, feature_col] == feature).sum()
    else:
        inside = df.loc[cluster_rows, feature_col].sum()
        parent_cluster = df.loc[parent_cluster_rows, feature_col].sum()
        all_with_feature = df[feature_col].sum()

    inside_size = cluster_rows.sum()
    # parent cluster

    parent_cluster_size = parent_cluster_rows.sum()
    outside = parent_cluster - inside

    expected_in = inside_size / parent_cluster_size * parent_cluster
    expected_out = parent_cluster - expected_in

    if expected_in < 5 or expected_out < 5:
        # logger.info(f"Skipping feature {feature_col}:{feature}. Incidence < 5.")
        return None

    score = (inside - expected_in) ** 2 + (outside - expected_out) ** 2
    if inside < expected_in:
        score = - score

    all_size = len(df)

    result = {'score': score,
              'inside': inside,
              'parent_cluster': parent_cluster,
              'inside_size': inside_size,
              'parent_cluster_size': parent_cluster_size,
              'all': all_with_feature,
              'all_size': all_size}
    return result


def print_scores(scores, top_x=None):
    if top_x is None:
        _print_scores(scores)
    else:
        _print_top_x_scores(scores, top_x)


def _print_top_x_scores(scores, top_x):
    """For each cluster, prints the top_x highest chi2 scores"""
    with open(config.DATA_FOLDER_PROCESSED / f"top_{top_x}_cluster_features.txt", 'w') as f:
        for cluster in scores:
            top_features = {}
            for feature_col in scores[cluster]:
                for feature in scores[cluster][feature_col]:
                    if feature == feature_col:
                        key = feature
                    else:
                        key = feature_col + "/" + feature
                    top_features[key] = scores[cluster][feature_col][feature]
                    top_features[key]["feature"] = feature
                    top_features[key]["feature_col"] = feature_col
                    top_features[key]["key"] = key

            top_features_sorted = [v for k, v in
                                   sorted(top_features.items(), key=lambda item: abs(item[1]['score']), reverse=True)]
            # print header
            if len(top_features_sorted) == 0:
                print("MISSING FEATURES: " + cluster)
                f.write("MISSING FEATURES : " + cluster + "\n")
                continue

            inside_size = top_features_sorted[0]['inside_size']
            parent_cluster_size = top_features_sorted[0]['parent_cluster_size']

            _print_cluster_header(cluster, f, inside_size, parent_cluster_size)

            for i in range(min(top_x, len(top_features_sorted))):
                inside = top_features_sorted[i]['inside']
                inside_size = top_features_sorted[i]['inside_size']
                parent_cluster = top_features_sorted[i]['parent_cluster']
                parent_cluster_size = top_features_sorted[i]['parent_cluster_size']
                all_with = top_features_sorted[i]['all']
                all_size = top_features_sorted[i]['all_size']

                to_print = (f"{top_features_sorted[i]['key']:42s} | {top_features_sorted[i]['score']:11.2f} |"
                            f"{inside:5.0f} / {inside_size:5.0f} = {inside / inside_size:6.4f} | "
                            f"{parent_cluster:5.0f} / {parent_cluster_size:5.0f} = {parent_cluster / parent_cluster_size:6.4f} | "
                            f"{all_with:5.0f} / {all_size:5.0f} = {all_with / all_size:6.4f}")
                print(to_print)
                f.write(to_print + "\n")


def _print_cluster_header(cluster, f, inside_size, parent_cluster_size):
    """Prints a nicely formatted cluster header"""
    to_print = f"{cluster} ({inside_size}/{parent_cluster_size} ~= {inside_size / parent_cluster_size:.2%})"
    f.write("=" * 75 + "\n")
    print("=" * 75)
    f.write(f"{to_print:^75} " + "\n")
    print(f"{to_print:^75} ")
    f.write("=" * 75 + "\n")
    print("=" * 75)


def _print_scores(scores):
    # sort by score per feature col
    for cluster in scores:
        for feature_col in scores[cluster]:
            scores[cluster][feature_col] = {k: v for k, v in sorted(scores[cluster][feature_col].items(),
                                                                    key=lambda item: abs(item[1]['score']),
                                                                    reverse=True)}

    # sort feature cols by name
    for cluster in scores:
        scores[cluster] = {k: v for k, v in sorted(scores[cluster].items(), key=lambda item: item[0])}

    # sort by cluster name
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[0])}

    with open(config.DATA_FOLDER_PROCESSED / "cluster_features.txt", 'w') as f:
        for cluster in scores:
            print_header = True
            if len(scores[cluster]) == 0:
                continue
            for feature_col in scores[cluster]:
                if len(scores[cluster][feature_col]) == 0:
                    continue
                for i, feature in enumerate(scores[cluster][feature_col]):
                    if print_header:
                        inside_size = scores[cluster][feature_col][feature]['inside_size']
                        parent_cluster_size = scores[cluster][feature_col][feature]['parent_cluster_size']
                        _print_cluster_header(cluster, f, inside_size, parent_cluster_size)
                        print_header = False

                    if i > 4:
                        break

                    # total visitors with some feature
                    # this cluster/parent/overall absolute + percentage
                    # expected

                    inside = scores[cluster][feature_col][feature]['inside']
                    inside_size = scores[cluster][feature_col][feature]['inside_size']
                    parent_cluster = scores[cluster][feature_col][feature]['parent_cluster']
                    parent_cluster_size = scores[cluster][feature_col][feature]['parent_cluster_size']
                    all_with = scores[cluster][feature_col][feature]['all']
                    all_size = scores[cluster][feature_col][feature]['all_size']

                    if feature == feature_col:
                        to_print = f"{str(feature):42} | "
                    else:
                        to_print = f"{str(feature_col):15s} | {str(feature):24s} | "

                    to_print += (f"{scores[cluster][feature_col][feature]['score']:11.2f} | "
                                 f"{inside:5.0f} / {inside_size:5.0f} = {inside / inside_size:6.4f} | "
                                 f"{parent_cluster:5.0f} / {parent_cluster_size:5.0f} = {parent_cluster / parent_cluster_size:6.4f} | "
                                 f"{all_with:5.0f} / {all_size:5.0f} = {all_with / all_size:6.4f}")

                    print(to_print)
                    f.write(to_print + "\n")
