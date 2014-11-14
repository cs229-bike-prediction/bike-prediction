import os

import numpy as np
import pandas as pd
from pandas.tseries.offsets import Day

from bike_prediction import get_coords as gc

FREQ = '1H'

def csvs(count):
    csv_files = os.listdir('./bike-rides/2013/')
    return map(lambda fn: open(os.path.join('./bike-rides/2013/', fn), 'rb'), csv_files[:count])

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
                      # 'mean_duration': r['Duration'].mean(),
                      # 'mean_distance': r.loc[:,('StartStation Id','EndStation Id')].apply(dist, axis=1).mean()
                      })

def get_rides(count):
    rides = None
    for csvfile in csvs(count):
        r = pd.read_csv(csvfile,
            quotechar='"',
            skipinitialspace=True,
            index_col='Start Date',
            parse_dates=True,
            dayfirst=True,
            usecols=['Rental Id', 'Start Date', 'Duration', 'StartStation Name', 'StartStation Id', 'EndStation Id'],
        )
        r = r[r['Duration'] > 0]
        r = r[r['StartStation Name'] != 'tabletop1']

        if rides is None:
            rides = r
        else:
            rides = rides.append(r)
    return rides

def read_data(rides):
    r = rides.groupby([pd.Grouper(freq=FREQ, level='Start Date'), 'StartStation Name']).apply(f).reset_index()
    return r.pivot(index='Start Date', columns='StartStation Name').fillna(value=0)

def add_weather_features(X):
    weather = pd.read_csv(open('./weather.csv', 'rb'),
            parse_dates=True,
            index_col='GMT',
            skipinitialspace=True,
            dtype={'Events': '|S32'})
    # weather = weather.asfreq(FREQ, method='pad')

    # Temperature
    temps = weather['Mean TemperatureF']
    temps = temps.asfreq(FREQ, method='pad')

    # Rain and other stuff
    # e = weather['Events'].astype(str)
    # print e.dtype
    # rain = np.char.find(e, 'Rain')
    e = np.asarray(weather['Events'], dtype=np.str_)

    rain_index    = np.char.find(e, 'Rain')
    fog_index     = np.char.find(e, 'Fog')
    snow_index    = np.char.find(e, 'Snow')
    hail_index    = np.char.find(e, 'Hail')
    thunder_index = np.char.find(e, 'Thunder')

    had_rain    = 1*np.int32(rain_index >= 0)
    had_fog     = 1*np.int32(fog_index >= 0)
    had_snow    = 3*np.int32(snow_index >= 0)
    had_hail    = 4*np.int32(hail_index >= 0)
    had_thunder = 4*np.int32(thunder_index >= 0)

    w_sum = had_rain + had_fog + had_snow + had_hail + had_thunder
    w = pd.Series(w_sum, name='weather', index=weather.index)
    w = w.asfreq(FREQ, method='pad')

    return pd.concat([X, temps, w], axis=1, join_axes=[X.index])

def add_historic_features(usage):
    s_d0 = usage['count']
    s_d1 = s_d0.shift(freq=Day(1))
    s_d2 = s_d1.shift(freq=Day(1))
    s_d3 = s_d2.shift(freq=Day(1))
    s_d7 = s_d0.shift(freq=Day(7))
    s_d14 = s_d0.shift(freq=Day(14))
    df_d1 = pd.DataFrame(s_d1.stack(), columns=['d1']).unstack()
    df_d2 = pd.DataFrame(s_d2.stack(), columns=['d2']).unstack()
    df_d3 = pd.DataFrame(s_d3.stack(), columns=['d3']).unstack()
    df_d7 = pd.DataFrame(s_d7.stack(), columns=['d7']).unstack()
    df_d14 = pd.DataFrame(s_d14.stack(), columns=['d14']).unstack()
    with_history = pd.concat([usage, df_d1, df_d2, df_d3, df_d7, df_d14], axis=1)
    return with_history

def filter_unknown_history(X):
    f = X[np.isfinite(X['d1'])]
    f = f[np.isfinite(f['d2'])]
    # f = f[np.isfinite(f['d3'])]
    f = f[np.isfinite(f['d7'])]
    f = f[np.isfinite(f['d14'])]
    f = f[np.isfinite(f['count'])]
    return f

def add_lat_long(X):
    gc.load_all_stations()
    f = X.join(gc.cache['locs_df'], on='StartStation Name')
    f = f[np.isfinite(f['lats'])]
    return f

def prepare_for_write(X):
    Z = X.copy()
    Z['Y'] = Z['count']
    del Z['count']
    del Z['StartStation Name']
    return Z

def save_sheet(df):
    df.to_excel('test.xls')
