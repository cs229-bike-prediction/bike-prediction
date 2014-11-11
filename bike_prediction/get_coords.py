import math
import xml.etree.ElementTree as ET

known_locs = {}
known_dists = {}

stations = ET.parse('cycle_locations.xml').getroot().findall('station')

def load_all_stations():
    for station in stations:
        r = (station.find('name').text,
                float(station.find('lat').text),
                float(station.find('long').text))
        known_locs[int(station.find('id').text)] = r

def get_station_location(station_id):
    if known_locs.has_key(station_id):
        return known_locs[station_id]
    else:
        for station in stations:
            if int(station.find('id').text) == station_id:
                r = (station.find('name').text,
                        float(station.find('lat').text),
                        float(station.find('long').text))
                known_locs[station_id] = r
                return r

def get_station_distances(id1, id2):
    if known_dists.has_key((id1, id2)):
        return known_dists[(id1, id2)]
    else:
        (name1, x1, y1) = get_station_location(id1)
        (name2, x2, y2) = get_station_location(id2)
        r = math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))
        known_dists[(id1, id2)] = r
        return r
