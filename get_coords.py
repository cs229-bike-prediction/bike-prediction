import math
import xml.etree.ElementTree as ET

def get_station_location(station_id):
    root = ET.parse('cycle_locations.xml').getroot()
    for station in root.findall('station'):
        if int(station.find('id').text) == station_id:
            return (station.find('name').text,
                    float(station.find('lat').text),
                    float(station.find('long').text))

def get_station_distances(id1, id2):
    (name1, x1, y1) = get_station_location(id1)
    (name2, x2, y2) = get_station_location(id2)
    return (name1, name2,
            math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2)))
