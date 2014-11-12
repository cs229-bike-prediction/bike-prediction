import math
import xml.etree.ElementTree as ET

known_locs = {}

stations = ET.parse('cycle_locations.xml').getroot().findall('station')

def load_all_stations():
    for station in stations:
        r = (station.find('name').text,
                float(station.find('lat').text),
                float(station.find('long').text))
        known_locs[int(station.find('id').text)] = r

def get_station_location(station_id):
    return known_locs[station_id]

def get_station_distances(id1, id2):
    (name1, x1, y1) = get_station_location(id1)
    (name2, x2, y2) = get_station_location(id2)
    return math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))
