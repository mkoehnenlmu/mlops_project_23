# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
import os 

from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    datasets = []

    for file in os.listdir("./data/raw/"):
        if ".csv" in file:
            data = pd.read_csv("./data/raw/"+file, sep = ",", quoting=False)
            datasets.append(data)

    data = pd.concat(datasets)
    data.shape


    ic_trains = data[~data['TIN'].isna() & data['TIN'].str.contains('IC')]
    ic_trains.shape


    ### Add some time features

    #Month
    ic_trains.loc[:,"MONTH"] = pd.to_datetime(ic_trains["TID"], format='%d.%m.%y').dt.month
    # Day of Week
    ic_trains.loc[:,"DOW"] = pd.to_datetime(ic_trains["TID"], format='%d.%m.%y').dt.dayofweek
    # Day
    ic_trains.loc[:,"DAY"] = pd.to_datetime(ic_trains["TID"], format='%d.%m.%y').dt.day
    # Hour
    ic_trains.loc[:,"HOUR"] = pd.to_datetime(ic_trains["TIT"], format='%H:%M').dt.hour
    ic_trains.loc[:,"MIN"] = pd.to_datetime(ic_trains["TIT"], format='%H:%M').dt.minute


    final_dataset = ic_trains[["TIN","TIR","TIM","TIRE","TIP","TSC","TSC","TAc",
                            "MONTH","DOW","DAY","HOUR","MIN"]]

    final_dataset.to_csv("./data/processed/data.csv")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
