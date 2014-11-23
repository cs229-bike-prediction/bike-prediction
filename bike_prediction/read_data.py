import os

import numpy as np
import pandas as pd
from pandas.tseries.offsets import Day, Hour

from bike_prediction import get_coords as gc

FREQ = '2H'

def csvs(count):
    csv_files = os.listdir('./bike-rides/')
    return map(lambda fn: open(os.path.join('./bike-rides/', fn), 'rb'), csv_files[:count])

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
        print csvfile
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

def filter_top_stations(rides, n):
    g = rides.groupby('StartStation Id').count()['Rental Id'].copy()
    g.sort(ascending=False)
    top_stations = g.index[:n].get_values()
    print top_stations
    return rides[rides['StartStation Id'].isin(top_stations)]

def filter_top_clusters(rides, n1,n2):
    g = rides.groupby('StartStation Id').count()['Rental Id'].copy()
    g.sort(ascending=False)
    top_stations = g.index[n1:n2].get_values()
    return top_stations

def add_cluster(X,c):
    nc=len(c)
    xn=len(X)
    labels=pd.Series(np.zeros(xn), name='label', index=X.index)
    X=pd.concat([X,labels],axis=1)
    for i in range(nc):
        X['label'][X['StartStation Id'].isin(c[i])]=i 
    return X

def read_data(rides):
    r = rides.groupby([pd.Grouper(freq=FREQ, level='Start Date'), 'StartStation Id']).apply(f).reset_index()
    return r.pivot(index='Start Date', columns='StartStation Id').fillna(value=0)
    # r = rides.resample('1H', how='count')
    # d = pd.DataFrame({'count': r['Rental Id']}, index=r.index)
    return d

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
    e = np.asarray(weather['Events'], dtype=np.str_)

    rain_index    = np.char.find(e, 'Rain')
    fog_index     = np.char.find(e, 'Fog')
    snow_index    = np.char.find(e, 'Snow')
    hail_index    = np.char.find(e, 'Hail')
    thunder_index = np.char.find(e, 'Thunder')

    had_rain    = 1*np.int32(rain_index >= 0)
    had_fog     = 1*np.int32(fog_index >= 0)
    had_snow    = 2*np.int32(snow_index >= 0)
    had_hail    = 2*np.int32(hail_index >= 0)
    had_thunder = 2*np.int32(thunder_index >= 0)

    w_sum = had_rain + had_fog + had_snow + had_hail + had_thunder
    w = pd.Series(w_sum, name='weather', index=weather.index)
    w = w.asfreq(FREQ, method='pad')

    return pd.concat([X, temps, w], axis=1, join_axes=[X.index])

def add_historic_features(usage):
    s_d0 = usage['count']
    s_h1 = s_d0.shift(freq=Hour(1))
    s_h25 = s_d0.shift(freq=Hour(25))
    s_d1 = s_d0.shift(freq=Day(1))
    s_d2 = s_d1.shift(freq=Day(1))
    s_d3 = s_d2.shift(freq=Day(1))
    s_d7 = s_d0.shift(freq=Day(7))
    s_d14 = s_d0.shift(freq=Day(14))
    s_d21 = s_d0.shift(freq=Day(21))
    s_d28 = s_d0.shift(freq=Day(28))
    df_h1 = pd.DataFrame(s_h1.stack(), columns=['h1']).unstack()
    df_h25 = pd.DataFrame(s_h25.stack(), columns=['h25']).unstack()
    df_d1 = pd.DataFrame(s_d1.stack(), columns=['d1']).unstack()
    df_d2 = pd.DataFrame(s_d2.stack(), columns=['d2']).unstack()
    df_d3 = pd.DataFrame(s_d3.stack(), columns=['d3']).unstack()
    df_d7 = pd.DataFrame(s_d7.stack(), columns=['d7']).unstack()
    df_d14 = pd.DataFrame(s_d14.stack(), columns=['d14']).unstack()
    df_d21 = pd.DataFrame(s_d21.stack(), columns=['d21']).unstack()
    df_d28 = pd.DataFrame(s_d28.stack(), columns=['d28']).unstack()
    with_history = pd.concat([usage, df_h1, df_h25, df_d1, df_d2, df_d3, df_d7, df_d14, df_d21, df_d28], axis=1)
    return with_history

def add_weekday(X):
    is_weekend = pd.DataFrame({'is_weekend': np.array((X.index.weekday==5) | (X.index.weekday==6), np.int)}, index=X.index)
    day = pd.DataFrame({'weekday': X.index.weekday}, index=X.index)
    return pd.concat([X, is_weekend, day], axis=1)

def filter_unknown_history(X):
    # f = X[np.isfinite(X['d1'])]
    # f = f[np.isfinite(f['d2'])]
    # f = f[np.isfinite(f['d3'])]
    # f = f[np.isfinite(f['d7'])]
    f = X[np.isfinite(X['d14'])]
    f = f[np.isfinite(f['count'])]
    f = f.fillna(0)
    return f

def add_lat_long(X):
    gc.load_all_stations()
    f = X.join(gc.cache['locs_df'], on='StartStation Id')
    f = f[np.isfinite(f['lats'])]
    return f

def prepare_for_write(X):
    Z = X.copy()
    Z['Y'] = Z['count']
    del Z['count']
    # del Z['StartStation Id']
    return Z

def save_sheet(df):
    df.to_excel('test.xls')
