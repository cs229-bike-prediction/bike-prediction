import numpy as np
import pandas as pd

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
    return pd.Series({'count': r['Rental Id'].count(),
                      'mean_duration': r['Duration'].mean(),
                      'mean_distance': r.loc[:,('StartStation Id','EndStation Id')].apply(dist, axis=1).mean()
                      })

def get_rides():
    rides = pd.read_csv(open('./bike-rides/1. Journey Data Extract 01Jan-05Jan13.csv', 'rb'),
                quotechar='"',
                skipinitialspace=True,
                index_col='Start Date',
                parse_dates=True,
                dayfirst=True)
    rides = rides[rides['Duration'] > 0]
    return rides

def read_data(rides):
    g = rides.groupby([pd.Grouper(freq='30T', level='Start Date'), 'StartStation Name']).apply(f).reset_index().set_index('Start Date')

    weather = pd.read_csv(open('./weather.csv', 'rb'), parse_dates=True, index_col='GMT')
    temps = weather['Mean TemperatureF']

    rw = pd.concat([g, temps['2013']], axis=1, join_axes=[g.index]).fillna(method='pad')    
    return rw
