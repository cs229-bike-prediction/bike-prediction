import numpy as np
import pandas as pd
from pandas.tseries.offsets import Day

from bike_prediction import get_coords as gc

def most_frequent_destination(dests):
    return dests.value_counts().idxmax()

def dist(r):
    try:
        id0 = r.iloc[0]
        id1 = r.iloc[1]
        r = gc.get_station_distances(id0, id1)
        return r
    except:
        return np.nan

def f(r):
    # Disable some of these to improve performance
    return pd.Series({'count': r['Rental Id'].count(),
                      'mean_duration': r['Duration'].mean(),
                      # 'mean_distance': r.loc[:,('StartStation Id','EndStation Id')].apply(dist, axis=1).mean()
                      })

def get_rides():
    rides = pd.read_csv(open('./bike-rides/1. Journey Data Extract 01Jan-05Jan13.csv', 'rb'),
                quotechar='"',
                skipinitialspace=True,
                index_col='Start Date',
                parse_dates=True,
                dayfirst=True,
                usecols=['Rental Id', 'Start Date', 'Duration', 'StartStation Name', 'StartStation Id', 'EndStation Id'],
            )
    rides = rides[rides['Duration'] > 0]
    return rides

def read_data(rides):
    r = rides.groupby([pd.Grouper(freq='30T', level='Start Date'), 'StartStation Name']).apply(f).reset_index()
    return r.pivot(index='Start Date', columns='StartStation Name').fillna(value=0)

def add_weather_features(X):
    weather = pd.read_csv(open('./weather.csv', 'rb'), parse_dates=True, index_col='GMT')
    temps = weather['Mean TemperatureF']

    return pd.concat([X, temps['2013']], axis=1, join_axes=[X.index]).fillna(method='pad')

def add_historic_features(usage):
    shifted = usage['count'].shift(freq=Day(1))
    prev_day = pd.DataFrame(shifted.stack(), columns=['prev_day']).unstack()
    with_history = pd.concat([usage, prev_day], axis=1)
    return with_history

def filter_unknown_history(X):
    f = X[np.isfinite(X['prev_day'])]
    return f[np.isfinite(f['count'])]

def save_sheet(df):
    df.to_excel('test.xls')
