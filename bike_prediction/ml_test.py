import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_records(csv):
    recs = np.recfromcsv(csv, names=True)
    return recs

def form_XY(recZ):
    station = recZ['startstation_id']
    h1 = np.array(recZ['h1'], np.float)
    h25 = np.array(recZ['h25'], np.float)
    d1 = np.array(recZ['d1'], np.float)
    d2 = np.array(recZ['d2'], np.float)
    d3 = np.array(recZ['d3'], np.float)
    d7 = np.array(recZ['d7'], np.float)
    d14 = np.array(recZ['d14'], np.float)
    meantemp = np.array(recZ['mean_temperaturef'], np.float)
    weather = np.array(recZ['weather'], np.float)
    is_weekend = np.array(recZ['is_weekend'], np.float)
    weekday = np.array(recZ['weekday'], np.float)
    lats = np.array(recZ['lats'], np.float)
    longs = np.array(recZ['longs'], np.float)

    X = np.array([h25, d1, d2, d3, d7, d14, weather, meantemp, weekday, is_weekend, lats, longs], np.float).T
    Y = recZ['y']
    locs = np.array([lats, longs], np.float).T

    return (station, X, Y, locs)

def divide_dataset(stations, X, Y, locs):
    train_len = int(stations.shape[0]*0.7)
    return {
        'tr_stations': stations[:train_len],
        'tr_X': X[:train_len,:],
        'tr_Y': Y[:train_len],
        'tr_locs': locs[:train_len],
        'tst_stations': stations[train_len:],
        'tst_X': X[train_len:,:],
        'tst_Y': Y[train_len:],
        'tst_locs': locs[train_len:]
    }

def predict_linreg(k, x_row):
    # Figure out what the closest cluster is
    # Now predict using that regression
    pass

def predict_all(dataset, k, models):
    tstX = dataset['tst_X']
    tstY = dataset['tst_Y']
    tstlen = tstX.shape[0]

    pred = np.zeros(tstlen)
    for i in range(tstlen):
        x_row = tstX[i,:]
        closest_cluster = k.predict(x_row)
        pred[i] = models[closest_cluster].predict(x_row)
    return mean_squared_error(pred, tstY)

def predict_on_station(dataset, k, linregs, poiss, station_id):
    X_station_a = dataset['tst_X'][dataset['tst_stations']==station_id,:]
    Y_station_a = dataset['tst_Y'][dataset['tst_stations']==station_id]
    loc_station_a = dataset['tst_locs'][dataset['tst_stations']==station_id]
    station_a_len = X_station_a.shape[0]

    linreg_predicts = np.zeros(station_a_len)
    # pois_predicts = np.zeros(station_a_len)
    for i in range(station_a_len):
        # linreg_predicts[i] = predict_linreg(k, X_station_a[i,:])
        x_row = X_station_a[i,:]
        closest_cluster = k.predict(x_row)
        linreg_predicts[i] = linregs[closest_cluster].predict(x_row)    
        # pois_predicts[i] = poiss[closest_cluster].predict(x_row)
    print mean_absolute_error(linreg_predicts,Y_station_a)
    # print mean_absolute_error(pois_predicts,Y_station_a)

    # Plot the behavior
    plt.plot(linreg_predicts, alpha=0.5, linewidth=2, color='g')
    plt.plot(Y_station_a, alpha=0.5, linewidth=2, color='b')
    plt.xlim(0,500)
    plt.show()
    # plt.plot(pois_predicts, alpha=0.5, linewidth=2, color='g')
    # plt.plot(Y_station_a, alpha=0.5, linewidth=2, color='b')
    # plt.xlim(0,500)
    # plt.show()

def plot_predicted_vs_real_zoom(predicted, real):
    zoom_l = int(len(predicted)*0.45)
    zoom_r = int(len(predicted)*0.5)

    f = plt.figure()
    f.set_figwidth(15)
    plt.subplot(121)
    plt.plot(predicted, alpha=0.5, linewidth=2, color='g')
    plt.plot(real, alpha=0.5, linewidth=2, color='r')
    plt.subplot(122)
    plt.plot(predicted[zoom_l:zoom_r], alpha=0.5, linewidth=2, color='g', label='Predicted')
    plt.plot(real[zoom_l:zoom_r], alpha=0.5, linewidth=2, color='r', label='Real')
    plt.legend()
    plt.show()

