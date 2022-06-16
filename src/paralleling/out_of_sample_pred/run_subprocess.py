import sys
import subprocess
import time
from tqdm import tqdm


if __name__ == "__main__":
    t1_start = time.process_time()
    original_ds = "Original"
    root_path = "/home/tpinho/IJGIS/Datasets/Australia_Election_2019"
    fs_method = "CFS"
    index_col = "INDEX"
    target_col = "TARGET"
    fold_col = "INDEX_FOLDS"
    brazil_datasets = [
        "Brazil_Election_2018_Sampled_dec0.3_prob0.1",
        "Brazil_Election_2018_Sampled_dec0.3_prob0.2",
        "Brazil_Election_2018_Sampled_dec0.3_prob0.3",
        "Brazil_Election_2018_Sampled_dec0.3_prob0.4",
        "Brazil_Election_2018_Sampled_dec0.3_prob0.5",
        "Brazil_Election_2018_Sampled_dec0.3_prob0.6",
        "Brazil_Election_2018_Sampled_dec0.3_prob0.7",
        "Brazil_Election_2018_Sampled_dec0.3_prob0.8",
        "Brazil_Election_2018_Sampled_dec0.3_prob0.9",
    ]

    us_corn_datasets = [
        "US_Corn_Yield_2016_Removed_ALABAMA",
        "US_Corn_Yield_2016_Removed_ARKANSAS",
        "US_Corn_Yield_2016_Removed_CALIFORNIA",
        "US_Corn_Yield_2016_Removed_COLORADO",
        "US_Corn_Yield_2016_Removed_DELAWARE",
        "US_Corn_Yield_2016_Removed_GEORGIA",
        "US_Corn_Yield_2016_Removed_IDAHO",
        "US_Corn_Yield_2016_Removed_ILLINOIS",
        "US_Corn_Yield_2016_Removed_INDIANA",
        "US_Corn_Yield_2016_Removed_IOWA",
        "US_Corn_Yield_2016_Removed_KANSAS",
        "US_Corn_Yield_2016_Removed_KENTUCKY",
        "US_Corn_Yield_2016_Removed_LOUISIANA",
        "US_Corn_Yield_2016_Removed_MARYLAND",
        "US_Corn_Yield_2016_Removed_MICHIGAN",
        "US_Corn_Yield_2016_Removed_MINNESOTA",
        "US_Corn_Yield_2016_Removed_MISSISSIPPI",
        "US_Corn_Yield_2016_Removed_MISSOURI",
        "US_Corn_Yield_2016_Removed_MONTANA",
        "US_Corn_Yield_2016_Removed_NEBRASKA",
        "US_Corn_Yield_2016_Removed_NEW JERSEY",
        "US_Corn_Yield_2016_Removed_NEW MEXICO",
        "US_Corn_Yield_2016_Removed_NEW YORK",
        "US_Corn_Yield_2016_Removed_NORTH CAROLINA",
        "US_Corn_Yield_2016_Removed_NORTH DAKOTA",
        "US_Corn_Yield_2016_Removed_OHIO",
        "US_Corn_Yield_2016_Removed_OKLAHOMA",
        "US_Corn_Yield_2016_Removed_PENNSYLVANIA",
        "US_Corn_Yield_2016_Removed_SOUTH CAROLINA",
        "US_Corn_Yield_2016_Removed_SOUTH DAKOTA",
        "US_Corn_Yield_2016_Removed_TENNESSEE",
        "US_Corn_Yield_2016_Removed_TEXAS",
        "US_Corn_Yield_2016_Removed_VIRGINIA",
        "US_Corn_Yield_2016_Removed_WEST VIRGINIA",
        "US_Corn_Yield_2016_Removed_WISCONSIN",
        "US_Corn_Yield_2016_Removed_WYOMING",
    ]

    us_corn_datasets = [
        "US_Corn_Yield_2016_Removed_Northeast",
        "US_Corn_Yield_2016_Removed_Southeast",
        "US_Corn_Yield_2016_Removed_Midwest",
        "US_Corn_Yield_2016_Removed_Southwest",
        "US_Corn_Yield_2016_Removed_West",
    ]

    australia_datasets = [
        "Australia_Election_2019_Sampled_dec0.05_prob0.1",
        "Australia_Election_2019_Sampled_dec0.05_prob0.2",
        "Australia_Election_2019_Sampled_dec0.05_prob0.3",
        "Australia_Election_2019_Sampled_dec0.05_prob0.4",
        "Australia_Election_2019_Sampled_dec0.05_prob0.5",
        "Australia_Election_2019_Sampled_dec0.05_prob0.6",
        "Australia_Election_2019_Sampled_dec0.05_prob0.7",
        "Australia_Election_2019_Sampled_dec0.05_prob0.8",
        "Australia_Election_2019_Sampled_dec0.05_prob0.9",
    ]

    procs = []
    for dataset_name in tqdm(australia_datasets):
        cmd = f'python out_sampled_train_pred.py {root_path} "{dataset_name}" {original_ds} {fs_method} {index_col} {fold_col} {target_col}'
        procs.append(subprocess.call(cmd, shell=True))
    exit_codes = [p.wait() for p in procs]
    t1_stop = time.process_time()
    print(f"time -- {(t1_start-t1_stop)/60}")
