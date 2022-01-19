import collections
import pickle
from pathlib import Path

import pandas as pd
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

import config
from src.data import _01_create_eventlog
from src.data import _02_convert_eventlog_to_sequences
from src.features import _01_create_behaviour_clusters
from src.features import _02_merge_behaviour_cluster_with_case_table
from src.features import enrich_case_table
from src.visualization import determine_features
from src.visualization.ExtractKPrototypesParams import ExtractKPrototypesParams


def has_purchase(seq, act=None):
    if act is None:
        act_dict = pd.read_parquet(config.DATA_FOLDER_PROCESSED / "sept_sequences_dict.parquet").to_dict(orient="dict")[
            'act_name']
        items = [key for key, value in act_dict.items() if value == "checkout:payment:pageview"]
        items += [key for key, value in act_dict.items() if value == "checkout:thankyou:pageview"]

    for i in range(len(seq) - len(act)):
        if seq[i:i + len(act)] == act:
            return True
    return False


# TODO: scaling -> relative for ONE person; not for ALL people

# TODO: scale every value to proportion for THAT PERSON -> 0.4 means 40% of time/sessions/w.e. spend on that activity/timeslot, etc
# TODO: to cluster: normalize proportions using mean + sd
# TODO; output: show the normalized proportion -> so we can easily compare between clusters
# TODO          show the absolute proportion -> so we can INTERPRET/WRITE PERSONAS
if __name__ == '__main__':
    # Settings
    N_CLUSTERS = 5
    # Create the dataset
    # merge eventlogs
    pd.read_csv = config.timed(pd.read_csv)
    pd.concat = config.timed(pd.concat)
    #
    result = pd.read_csv(config.DATA_FOLDER_RAW / "SGH_w42_export_v4.zip")
    # sept = pd.read_csv(config.DATA_FOLDER_RAW / "SGH_sept.zip")
    #
    # print(data.columns, sept.columns)
    #
    # result = pd.concat([data, sept], axis=0)
    result.to_csv(config.DATA_FOLDER_RAW / "w42_data.csv")
    # run analysis
    _01_create_eventlog.create_eventlog(config.DATA_FOLDER_RAW / "w42_data.csv", output_prefix="w42", read_parquet=False, store_parquet=config.DATA_FOLDER_RAW / "data.parquet")
    _02_convert_eventlog_to_sequences.convert_eventlog(input=config.DATA_FOLDER_INTERIM / "w42_eventlog.parquet", ngrams=0, output_prefix="w42")
    case_table, scaler, purchase = enrich_case_table.run(
        sequence_file=config.DATA_FOLDER_PROCESSED / "w42_all_sequences.parquet",
        sequence_dict_file=config.DATA_FOLDER_PROCESSED / "w42_sequences_dict.parquet",
        case_table_file = config.DATA_FOLDER_INTERIM / "w42_case_table.parquet",
        eventlog_file = config.DATA_FOLDER_INTERIM / "w42_eventlog.parquet",
        store=config.DATA_FOLDER_PROCESSED / "w42_enriched_case_table",
        standardize=True,
        filter_seq_length=5)
    # keep_cols = [col_name for col_name in case_table.columns if col_name.endswith("standardized")]
    # keep_cols += ['city', 'area', 'country', 'device', 'os', 'ltt', "purchase_indicator"]
    # keep_cols = [col for col in case_table.columns if
    #              col in keep_cols]  # preserve original order which is necessary for the scaler
    # X = case_table.dropna()[keep_cols]
    # X = X.sample(n=50000)
    # drop_cols = []
    # X = X.drop(columns=drop_cols)
    # X["mon_standardized"] = X["mon_standardized"] / 7
    # X["tue_standardized"] = X["tue_standardized"] / 7
    # X["wed_standardized"] = X["wed_standardized"] / 7
    # X["thu_standardized"] = X["thu_standardized"] / 7
    # X["fri_standardized"] = X["fri_standardized"] / 7
    # X["sat_standardized"] = X["sat_standardized"] / 7
    # X["sun_standardized"] = X["sun_standardized"] / 7
    # print(X.columns)

    # # Verify no NAs
    # print("Has NAs:", X.isna().sum().sum() != 0)
    # print("NAs in rows:", str(X.columns[X.isnull().any()].tolist()))

    # kp = KPrototypes(n_clusters=N_CLUSTERS, init="huang", random_state=42, n_jobs=-2, verbose=True)
    # categorical = [X.columns.get_loc(name) for name in X.select_dtypes(exclude="number")]
    # clusters = config.timed(kp.fit)(X, categorical=categorical)
    # with open(Path("models/kp_data_50k_sample_5plus.pickle"), "wb") as f:
    #     pickle.dump((kp, X.columns, categorical), f)

    # extractor = ExtractKPrototypesParams(features=X.columns, categoricals=categorical)
    # extracted = extractor.extract(kp)
    # pred = kp.predict(X, categorical=categorical)
    # purchase = X['purchase_indicator']
    # extracted_with_rescale = extractor.extract_with_rescale(kp, scaler=scaler,
    #                                                         dont_inv_trans=["purchase_indicator"], pred=pred, purchase=purchase)
    #
    # extracted_with_rescale.to_excel(
    #     config.DATA_FOLDER_PROCESSED / "kp_data_50k_sample_5plus_prototypes.xlsx")
    # #
    # # Filter the users on having a purchase
    # all_sequences = _01_create_behaviour_clusters.get_sequences()
    # filtered = _02_convert_eventlog_to_sequences.filter_sequence_length(all_sequences, min_length=1) # note original analyses was done with min_length = 1 -> 15k purchases
    # filtered = _01_create_behaviour_clusters.filter_sequences(filtered, on='purchase')
    #
    # # Create the clustering
    # clusters, tree = _01_create_behaviour_clusters. \
    #     create_and_extract_clusters(sample=None,
    #                                 override_random=42,
    #                                 sequences=filtered,
    #                                 min_cluster_size=100,
    #                                 output_tree=config.REPORT_FOLDER / "purchase_clusters_analyzed.json",
    #                                 output_file=config.DATA_FOLDER_PROCESSED / "purchase_clusters_analyzed.parquet",
    #                                 exclude=[])

    # (Optional) Read an already performed clustering
    # clusters = _01_create_behaviour_clusters.get_clusters(config.DATA_FOLDER_PROCESSED / "all_purchase_clusters.parquet")
    # case_table = enrich_case_table.retrieve(config.DATA_FOLDER_PROCESSED / "case_table_with_features_scaled.parquet")
    # # # Merge with case_table
    # merged = _02_merge_behaviour_cluster_with_case_table.merge_behavioural(clusters, case_table)
    # # # Store
    # merged.to_csv(config.REPORT_FOLDER / "all_purchase_clusters_case_table.csv")
    # # # Determine & write feature scores
    # feature_scores = determine_features.get_feature_scores(merged.drop(columns=["sequence", "visitor_id", "city"]))
    # determine_features.print_scores(feature_scores)
    # determine_features.print_scores(feature_scores, top_x=20)

    # Get enriched case table
    # case_table, scaler, purchase = enrich_case_table.run(store=True, standardize=True, filter_seq_length=5)
    # # case_table, scaler = enrich_case_table.retrieve(with_scaler=True)

    # use only standardized features
    keep_cols = [col_name for col_name in case_table.columns if col_name.endswith("standardized")]
    keep_cols += ['city', 'area', 'country', 'device', 'os', 'ltt', "purchase_indicator"]
    keep_cols = [col for col in case_table.columns if
                 col in keep_cols]  # preserve original order which is necessary for the scaler
    X = case_table.dropna()[keep_cols]
    purchase = purchase[~case_table.isna().any(axis=1)]

    # Drop features (discussion 24/11)
    # drop_cols = ["7d_before_labor_day_standardized",
    #              "3d_before_labor_day_standardized",
    #              "on_labor_day_standardized",
    #              "3d_after_labor_day_standardized",
    #              "7d_after_labor_day_standardized",
    #              "7d_before_new_york_fashion_week_standardized",
    #              "3d_before_new_york_fashion_week_standardized",
    #              "on_new_york_fashion_week_standardized",
    #              "3d_after_new_york_fashion_week_standardized",
    #              "7d_after_new_york_fashion_week_standardized",
    #              "7d_before_london_fashion_week_standardized",
    #              "3d_before_london_fashion_week_standardized",
    #              "on_london_fashion_week_standardized",
    #              "3d_after_london_fashion_week_standardized",
    #              "7d_after_london_fashion_week_standardized",
    #              "7d_before_milan_fashion_week_standardized",
    #              "3d_before_milan_fashion_week_standardized",
    #              "on_milan_fashion_week_standardized",
    #              "3d_after_milan_fashion_week_standardized",
    #              "7d_after_milan_fashion_week_standardized",
    #              "7d_before_paris_fashion_week_spring/summer_2021_standardized",
    #              "3d_before_paris_fashion_week_spring/summer_2021_standardized",
    #              "on_paris_fashion_week_spring/summer_2021_standardized",
    #              "m20help:pageview_standardized",
    #              "end of session_standardized",
    #              "checkout:delivery:pageview_standardized",
    #              "checkout:gateway:pageview_standardized",
    #              "checkout:payment:pageview_standardized",
    #              "checkout:thankyou:pageview_standardized",
    #              "purchase_indicator"]
    # X = X.drop(columns=drop_cols)

    # X = X.drop(columns=["purchase_indicator"])
    X["mon_standardized"] = X["mon_standardized"] / 7
    X["tue_standardized"] = X["tue_standardized"] / 7
    X["wed_standardized"] = X["wed_standardized"] / 7
    X["thu_standardized"] = X["thu_standardized"] / 7
    X["fri_standardized"] = X["fri_standardized"] / 7
    X["sat_standardized"] = X["sat_standardized"] / 7
    X["sun_standardized"] = X["sun_standardized"] / 7

    print(X.columns)

    # Verify no NAs
    print("Has NAs:", X.isna().sum().sum() != 0)
    print("NAs in rows:", str(X.columns[X.isnull().any()].tolist()))

    # Create purchase model
    # kp_purchase = KPrototypes(n_clusters=N_CLUSTERS, init="huang", random_state=42, n_jobs=-1, verbose=True)
    # X_purchase = X[X['purchase_indicator'] == 1]
    # # determine categorical features
    # categorical = [X_purchase.columns.get_loc(name) for name in X_purchase.select_dtypes(exclude="number")]
    # # Fit
    # config.timed(kp_purchase.fit)(X_purchase, categorical=categorical)
    # # Get and store model
    # with open(Path("models/kp_purchases_standardized.pickle"), "wb") as f:
    #     pickle.dump((kp_purchase, X_purchase.columns, categorical), f)

    '''
    # Create overall model
    kp = KPrototypes(n_clusters=N_CLUSTERS, init="huang", random_state=42, n_jobs=-1, verbose=True)
    categorical = [X.columns.get_loc(name) for name in X.select_dtypes(exclude="number")]
    clusters = config.timed(kp.fit)(X, categorical=categorical)
    with open(Path("models/kp_standardized_full_v2.pickle"), "wb") as f:
        pickle.dump((kp, X.columns, categorical), f)
    '''
    # params = {i: {k:v for k,v in zip(X.columns[6:], kp.cluster_centroids_[0].tolist()[i])} for i in range(10)}
    # for k, v in params.items():
    #     for feature, score in sorted(v.items(), key=lambda x: x[0]):
    #         print(f"Cluster {k:2d}: {feature:65s}={score:.4f}")


    with open(enrich_case_table.SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    # with open(Path("models/kp_purchases_standardized.pickle"), "rb") as f:
    #     kp_purchases_standardized, column_names, categorical = pickle.load(f)
    #
    # extractor = ExtractKPrototypesParams(features=column_names, categoricals=categorical)
    # purchases_extracted = extractor.extract(kp_purchases_standardized)
    # purchases_extracted_with_rescale = extractor.extract_with_rescale(kp_purchases_standardized, scaler=scaler,
    #                                                                   dont_inv_trans=["purchase_indicator"])
    #
    # purchases_extracted_with_rescale.to_excel(
    #     config.DATA_FOLDER_PROCESSED / "kp_purchases_standardized_prototypes.xlsx")

    with open(Path("models/kp_standardized_full_v2.pickle"), "rb") as f:
        kp_standardized, column_names, categorical = pickle.load(f)

    extractor = ExtractKPrototypesParams(features=column_names, categoricals=categorical)
    extracted = extractor.extract(kp_standardized)
    pred = kp_standardized.predict(X, categorical=categorical)
    #pred = kp_standardized.predict(X.drop(columns=["purchase_indicator"]), categorical=categorical)
    extracted_with_rescale = extractor.extract_with_rescale(kp_standardized, scaler=scaler, dont_inv_trans=["purchase_indicator"],
                                                             pred=pred, purchase=purchase)
    # Store the feature values per cluster
    extracted_with_rescale_t = extracted_with_rescale.transpose().reset_index()
    extracted_with_rescale_t = extracted_with_rescale_t.rename(columns={"index": "cluster_details"})
    extracted_with_rescale_t['cluster'] = extracted_with_rescale_t['cluster_details'].str[8]
    cols = list(extracted_with_rescale_t.columns)
    cols = [cols[-1]] + cols[:-1]
    extracted_with_rescale_t = extracted_with_rescale_t[cols]
    extracted_with_rescale_t['cluster'] = extracted_with_rescale_t['cluster'].fillna(999)
    extracted_with_rescale_t_non_stand = extracted_with_rescale_t.iloc[1:len(extracted_with_rescale_t)-2,:].iloc[::2, :]

    drop_columns = ["city","area","country","device","os","ltt","purchase_indicator"]
    extracted_with_rescale_t_non_stand = extracted_with_rescale_t_non_stand.drop(drop_columns, axis=1)


    extracted_with_rescale_t_stand = extracted_with_rescale_t.iloc[:len(extracted_with_rescale_t)-2,:].iloc[::2, :]
    extracted_with_rescale_t_stand = extracted_with_rescale_t_stand.append(extracted_with_rescale_t.iloc[len(extracted_with_rescale_t)-2:,:])

    extracted_with_rescale_t_stand.to_csv(config.DATA_FOLDER_PROCESSED / "kp_all_standardized_2plus_v2_prototypes.csv", index= False)
    extracted_with_rescale_t_non_stand.to_csv(config.DATA_FOLDER_PROCESSED / "kp_all_2plus_v2_prototypes.csv", index= False)

    # Store the cluster predictions per customer

    X_predicted_clusters = X.reset_index().join(pd.DataFrame(pred, columns=['cluster']))
    X_predicted_clusters.to_csv(config.DATA_FOLDER_PROCESSED / "casetable_kprot_2plus_v2_prototypes.csv", index=False)
