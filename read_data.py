import numpy as np
import pandas as pd

def most_frequent_destination(dests):
    return dests.value_counts().idxmax()

def f(r):    
    return pd.Series({'count': r['Rental Id'].count(),
                      'mean_duration': r['Duration'].mean(),
                      'most_common': most_frequent_destination(r['EndStation Name'])
                      })

def read_data():
    rides = pd.read_csv(open('./bike-rides/1. Journey Data Extract 01Jan-05Jan13.csv', 'rb'),
                quotechar='"',
                skipinitialspace=True,
                index_col='Start Date',
                parse_dates=True,
                dayfirst=True)
    rides = rides.sort_index()
    rides = rides[rides['Duration'] > 0]
    g = rides.groupby([pd.Grouper(freq='H', level='Start Date'), 'StartStation Name']).apply(f).reset_index().set_index('Start Date')

    weather = pd.read_csv(open('./weather.csv', 'rb'), parse_dates=True, index_col='GMT')
    temps = weather['Mean TemperatureF']

    rw = pd.concat([g, temps['2013']], axis=1, join_axes=[g.index]).fillna(method='pad')    
    return rw
