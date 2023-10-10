import pandas as pd


# Function that takes in a dataframe and returns a dataframe
# with the following features:
def build_features(ic_trains: pd.DataFrame):

    # Add some time features
    # Month
    ic_trains.loc[:, "MONTH"] = pd.to_datetime(ic_trains["TID"],
                                               format='%d.%m.%y').dt.month
    # Day of Week
    ic_trains.loc[:, "DOW"] = pd.to_datetime(ic_trains["TID"],
                                             format='%d.%m.%y').dt.dayofweek
    # Day
    ic_trains.loc[:, "DAY"] = pd.to_datetime(ic_trains["TID"],
                                             format='%d.%m.%y').dt.day
    # Hour
    ic_trains.loc[:, "HOUR"] = pd.to_datetime(ic_trains["TIT"],
                                              format='%H:%M').dt.hour
    ic_trains.loc[:, "MIN"] = pd.to_datetime(ic_trains["TIT"],
                                             format='%H:%M').dt.minute

    # convert TIRE column to one hot encoding
    ic_trains = pd.concat([ic_trains, ic_trains["TIRE"].str.get_dummies()],
                          axis=1)

    # create binary variable whether string in column TIN contains IC or ICE
    ic_trains["ICE"] = ic_trains["TIN"].str.contains("ICE").astype(int)

    # save the number at the end of the TIN column as a new column
    ic_trains = ic_trains.reset_index()
    ic_trains["TNR"] = [int(ic_trains.loc[i, "TIN"][3:])
                        if ic_trains.loc[i, "ICE"] == 1
                        else int(ic_trains.loc[i, "TIN"][2:])
                        for i in range(len(ic_trains["TIN"]))]
    # TIM == arrival or departure
    ic_trains["ARR"] = (ic_trains["TIM"] == "arr").astype(int)

    ic_trains["TIP"] = [int(i) if isinstance(i, int) else 0
                        for i in ic_trains["TIP"]]

    return ic_trains


def build_features_new(data:pd.DataFrame):
    
    # Convert distance features
    data["DISTANCE"] = [int(x[0]) for x in (data["DEP_TIME_BLK"].str.split("-"))]

    # convert columns to one hot encoding
    data = pd.concat([data, data["CARRIER_NAME"].str.get_dummies()],
                        axis=1)
    data = pd.concat([data, data["DEPARTING_AIRPORT"].str.get_dummies()],
                        axis=1)
    data = pd.concat([data, data["PREVIOUS_AIRPORT"].str.get_dummies()],
                        axis=1)

    return data
