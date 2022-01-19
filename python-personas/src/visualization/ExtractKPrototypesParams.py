import pandas as pd


class ExtractKPrototypesParams:

    def __init__(self, features, categoricals):
        self.features = features
        self.categoricals = categoricals
        self.categorical_features = [features[i] for i in categoricals]
        self.numeric_features = [features[i] for i in range(len(features)) if i not in categoricals]

    def extract(self, kp):
        n_cluster = kp.n_clusters

        df = pd.DataFrame(index=self.features)  # empty dataframe
        for n in range(n_cluster):
            assert len(self.categorical_features) == len(kp.cluster_centroids_[1][n])
            assert len(self.numeric_features) == len(kp.cluster_centroids_[0][n])
            # create feature_name -> score dict
            params = {self.numeric_features[i]: kp.cluster_centroids_[0][n][i]
                      for i in range(len(self.numeric_features))}
            cat_params = {self.categorical_features[i]: kp.cluster_centroids_[1][n][i]
                          for i in range(len(self.categorical_features))}
            params.update(cat_params)
            column = pd.DataFrame.from_dict(params, orient="index", columns=["cluster"])
            df["cluster_" + str(n)] = column['cluster']

        return df

    def extract_with_rescale(self, kp, scaler=None, dont_inv_trans=None, pred=None, purchase=None):
        df = pd.DataFrame(index=list(self.features) + ['size', 'purchase_perc'])  # empty dataframe
        if dont_inv_trans is None:
            dont_inv_trans = []
        col_names = [name for name in self.numeric_features if name not in dont_inv_trans]

        for n in range(kp.n_clusters):
            cat = 0
            num = 0
            params = {}
            if pred is not None and purchase is not None:
                selection = (pred == n)
                size = selection.sum()
                purchase_percentage = purchase[selection].sum() / selection.sum()
            for i, feature in enumerate(self.features):
                if i in self.categoricals:
                    params[feature] = kp.cluster_centroids_[1][n][cat]
                    cat += 1
                else:
                    params[feature] = kp.cluster_centroids_[0][n][num]
                    num += 1

            column = pd.DataFrame.from_dict(params, orient="index", columns=["cluster"])
            df["cluster_" + str(n) + "_stand"] = column["cluster"]
            if scaler is not None:
                # we do some transposing because features are now horizontal
                temp = pd.DataFrame(params, index=["cluster"]).drop(columns=self.categorical_features + dont_inv_trans)
                de_trans = scaler.inverse_transform(temp)
                de_trans = pd.DataFrame(de_trans, columns=col_names, index=["cluster"])
                if pred is not None and purchase is not None:
                    de_trans['size'] = size
                    de_trans['purchase_perc'] = purchase_percentage
                de_trans = de_trans.transpose()
                df["cluster_" + str(n)] = de_trans["cluster"]

        df['mean'] = pd.Series(scaler.mean_, index=col_names)
        df['std'] = pd.Series(scaler.scale_, index=col_names)
        return df
