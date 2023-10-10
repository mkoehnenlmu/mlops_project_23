# -*- coding: utf-8 -*-
import logging
import os
import pandas as pd

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import src.features.build_features as bf


def make_old_dataset(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from
    (../input_filepath) into cleaned data ready to be analyzed
    (saved in ../output_filepath).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    datasets = []

    for file in os.listdir(input_filepath):
        if ".csv" in file:
            data = pd.read_csv(
                os.path.join(input_filepath, file), sep=",", quoting=False
            )
            datasets.append(data)

    data = pd.concat(datasets)

    # filter out trains that are not ICE or IC
    ic_trains = data[~data["TIN"].isna() & data["TIN"].str.contains("IC")]

    feature_data = bf.build_features(ic_trains)

    final_dataset = feature_data.drop(
        ["TIN", "TIRE", "TIL", "TSC", "TA", "TIR", "index", "TID", "TIT", "TIM", "TAA"],
        axis=1,
    )

    final_dataset.to_csv("./data/processed/data.csv", index=False)


def make_new_dataset(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from
    (../input_filepath) into cleaned data ready to be analyzed
    (saved in ../output_filepath).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    train = pd.read_csv(input_filepath + "train.csv")

    train_s = train.sample(500000, random_state=42)
    train_inverse = train.drop(train_s.index)
    train_test = train_inverse.sample(100000)

    logger.info("Computing train features")
    feature_data_train = bf.build_features_new(train)
    feature_data_train = bf.build_features_new(train_test)

    final_dataset_train = feature_data_train.drop(
        ["CARRIER_NAME", "DEPARTING_AIRPORT", "DEP_TIME_BLK", "PREVIOUS_AIRPORT"],
        axis=1,
    )
    final_dataset_test = train_test.drop(
        ["CARRIER_NAME", "DEPARTING_AIRPORT", "DEP_TIME_BLK", "PREVIOUS_AIRPORT"],
        axis=1,
    )

    final_dataset_train.to_csv(output_filepath + "train_sample.csv", index=False)
    final_dataset_test.to_csv(output_filepath + "test_sample.csv", index=False)


def main(input_filepath, output_filepath, data: str = "new"):
    if data == "new":
        make_new_dataset(input_filepath, output_filepath)
    else:
        make_old_dataset(input_filepath, output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main("data/raw/", "data/processed/", "new")
