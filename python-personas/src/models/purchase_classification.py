from itertools import chain

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

import config
from data import _02_convert_eventlog_to_sequences
from features import _01_create_behaviour_clusters
from src.features import enrich_case_table

SEQUENCES_FILE = config.DATA_FOLDER_PROCESSED / "sept_all_sequences.parquet"
CASE_TABLE = config.DATA_FOLDER_INTERIM / "sept_case_table.parquet"


def get_dataset():
    seq = pd.read_parquet(SEQUENCES_FILE)
    print(f"Seq shape: {seq.shape}")
    act_dict = _01_create_behaviour_clusters.get_activity_dict()
    print("Creating enriched case table.")
    case = enrich_case_table.run(False)
    merged = seq.merge(case, on="user_id")
    merged = _02_convert_eventlog_to_sequences.filter_sequence_length(merged, min_length=4)
    vectorizer = CountVectorizer(lowercase=False, analyzer=lambda x: x, vocabulary=act_dict.keys())
    counts = vectorizer.fit_transform(merged['sequence'].tolist())
    counts = pd.DataFrame(counts.todense(), columns=list(act_dict.values()))
    merged = merged.merge(counts, left_index=True, right_index=True)
    purchase = _01_create_behaviour_clusters.filter_sequences(merged, on='purchase')
    purchase['purchase'] = 1
    near_purchase = _01_create_behaviour_clusters.filter_sequences(merged, on='near_purchase')
    near_purchase['purchase'] = 0

    merged = pd.concat([purchase, near_purchase])
    print(merged.shape, (merged['purchase'] == 1).sum(), (merged['purchase'] == 0).sum())
    return merged


def get_split(df):
    drop = ['sequence', 'visitor_id', 'timestamp', 'purchase', 'checkout:thankyou:pageview', 'city', 'area', 'country',
            "m20help:pageview", 'searchdisplay:pageview', 'order status:pageview', 'checkout:gateway:pageview', 'checkout:payment:pageview']
    X = df.drop(drop, axis=1)
    X = pd.get_dummies(X, columns=["device", "os", "ltt"])
    X = X.drop(['os_gnu/linux'], axis=1)
    names = X.columns
    y = df['purchase']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test, names


if __name__ == '__main__':
    # df = get_dataset()
    # df.to_parquet(config.DATA_FOLDER_PROCESSED / "df_for_classification.parquet")
    df = pd.read_parquet(config.DATA_FOLDER_PROCESSED / "df_for_classification.parquet")
    X_train, X_test, y_train, y_test, feature_names = get_split(df)

    lr = LogisticRegression(max_iter=1000, penalty="elasticnet", solver='saga', C=0.5, l1_ratio=0.5)



    lr.fit(X_train, y_train)

    lr.score(X_train, y_train)

    lr.score(X_test, y_test)

    coef_dict = {}
    for coef, feat in zip([x for tmp in lr.coef_ for x in tmp], feature_names):
        coef_dict[feat] = coef

    for key, val in sorted(coef_dict.items(), key=lambda item: (item[1]), reverse=True):
        print(f"{key:40s} = {val:>10.5f}")

    ab = AdaBoostClassifier(n_estimators=100)
    ab.fit(X_train, y_train)
    ab.score(X_train, y_train)
    ab.score(X_test, y_test)
    print(ab.feature_importances_)
    for coef, feat in zip([x for x in ab.feature_importances_], feature_names):
        coef_dict[feat] = coef

    for key, val in sorted(coef_dict.items(), key=lambda item: (item[1]), reverse=True):
        print(f"{key:40s} = {val:>10.5f}")