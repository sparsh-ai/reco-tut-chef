
import os
import sys
import yaml
import pandas as pd

from src.dataset import SessionDataset

params = yaml.safe_load(open("params.yaml"))["prepare_30music"]

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython {dataprep}.py {data-source-filepath}\n")
    sys.exit(1)

filter_last_months = params['filter_last_months']
filter_topk = params['filter_topk']

data_source_path = str(sys.argv[1])
data_target_path = os.path.join("data", "silver", "30music")
data_target_path_train = os.path.join(data_target_path, 'train.parquet.snappy')
data_target_path_test = os.path.join(data_target_path, 'test.parquet.snappy')


def prepare_data():
    df = pd.read_parquet(data_source_path)
    sess_ds = SessionDataset(df)

    sess_ds.filter_by_time(filter_last_months)
    sess_ds.convert_to_sequence(filter_topk)
    stats = sess_ds.get_stats()
    print(stats)
    sess_ds.last_session_out_split()
    print("Train sessions: {} - Test sessions: {}".format(len(sess_ds.train), len(sess_ds.test)))
    sess_ds.train.to_parquet(data_target_path_train, compression='snappy')
    sess_ds.test.to_parquet(data_target_path_test, compression='snappy')


os.makedirs(data_target_path, exist_ok=True)

prepare_data(data_source_path,
             filter_last_months, filter_topk,
             data_target_path_train, data_target_path_test)