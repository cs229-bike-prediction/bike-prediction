def get_coords(station):
    import xml.etree.ElementTree as ET
    tree = ET.parse('stations.kml')
    root = tree.getroot()
    #start = 'Tower Gateway'
    it = root.iter(None);
    while True:
        try:
            value = it.next()
            if station in value.text:
                value = it.next();
                value = it.next();
                value = it.next();
                coords = value.text.strip();
                coords = coords.split(',',2);
                x_coord = coords[0];
                y_coord = coords[1];
                return x_coord,y_coord;
        except StopIteration:
            break
