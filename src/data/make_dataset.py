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

    ic_trains = data[~data['TIN'].isna() & data['TIN'].str.contains('IC')]

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

    # convert TIRE column to one hot encoding
    ic_trains = pd.concat([ic_trains, ic_trains["TIRE"].str.get_dummies()], axis=1)

    # create a binary variable whether the string in column TIN contains IC or ICE
    ic_trains["ICE"] = ic_trains["TIN"].str.contains("ICE").astype(int)

    # save the number at the end of the TIN column as a new column
    ic_trains = ic_trains.reset_index()
    ic_trains["TNR"] = [int(ic_trains.loc[i, "TIN"][3:]) if ic_trains.loc[i, "ICE"] == 1
                   else int(ic_trains.loc[i, "TIN"][2:])
                    for i in range(len(ic_trains["TIN"]))]
    
    # TIM == arrival or departure
    ic_trains["ARR"] = (ic_trains["TIM"] == "arr").astype(int)

    ic_trains["TIP"] = [int(i) if type(i) == int else 0 for i in ic_trains["TIP"]]


    final_dataset = ic_trains.drop(["TIN", "TIRE", "TIL",
                                    "TSC", "TA", "TIR", "index",
                                    "TID", "TIT", "TIM", "TAA"], axis = 1)
    
    final_dataset.to_csv("./data/processed/data.csv", index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    #main()
    main(["./data/raw/", "hello"])
