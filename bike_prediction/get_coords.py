import math
import xml.etree.ElementTree as ET

import pandas as pd

known_locs = {}
cache = {'locs_df': None}

stations = ET.parse('cycle_locations.xml').getroot().findall('station')

def load_all_stations():
    ids   = []
    names = []
    lats  = []
    longs = []

    for station in stations:
        ident = int(station.find('id').text)
        name  = station.find('name').text
        lati  = float(station.find('lat').text)
        longi = float(station.find('long').text)
        r = (name, lati, longi)
        known_locs[name] = r

        ids.append(ident)
        names.append(name)
        lats.append(lati)
        longs.append(longi)

    cache['locs_df'] = pd.DataFrame({'lats': lats, 'longs': longs}, index=ids)

def get_station_location(station_name):
    return known_locs[station_name]

def get_station_distances(name1, name2):
    (name1, x1, y1) = get_station_location(name1)
    (name2, x2, y2) = get_station_location(name2)
    return math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))
