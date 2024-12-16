##Download the updated Bus stops data in Baku
import numpy as np
import requests
import geopandas as gpd
from shapely.geometry import Point, Polygon


def download_bus_stops_for_baku():
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json];
    node
      ["highway"="bus_stop"]
      
      (40.300, 49.700,40.590, 50.400);  // Baku bounding box
    out body;
    """
    try:
        response = requests.get(overpass_url, params={'data': overpass_query})
        response.raise_for_status()
        data = response.json()
    finally:
        response.close()
    
    bus_stops = []
    for element in data['elements']:
        if element['type'] == 'node':
            bus_stops.append({
                'id': element['id'],
                'lat': element['lat'],
                'lon': element['lon'],
                'name': element.get('tags', {}).get('name', 'Unnamed')
            })
    
    # Create a GeoDataFrame using public APIs
    geometry = [Point(stop['lon'], stop['lat']) for stop in bus_stops]
    bus_stops_gdf = gpd.GeoDataFrame(bus_stops, geometry=geometry)
    bus_stops_gdf.set_crs("EPSG:4326", inplace=True)
    
    return bus_stops_gdf

###Download the updated Metro stations in Baku
def download_metro_stations_for_baku():
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json];
    node
      ["railway"="station"]
      ["station"="subway"]
      (40.300, 49.700, 40.590, 50.400);  // Baku bounding box
    out body;
    """
    try:
        response = requests.get(overpass_url, params={'data': overpass_query})
        response.raise_for_status()
        data = response.json()
    finally:
        response.close()
    
    metro_stations = []
    for element in data['elements']:
        if element['type'] == 'node':
            metro_stations.append({
                'id': element['id'],
                'lat': element['lat'],
                'lon': element['lon'],
                'name': element.get('tags', {}).get('name', 'Unnamed')
            })
    
    # Create a GeoDataFrame using public APIs
    geometry = [Point(station['lon'], station['lat']) for station in metro_stations]
    metro_stations_gdf = gpd.GeoDataFrame(metro_stations, geometry=geometry)
    metro_stations_gdf.set_crs("EPSG:4326", inplace=True)
    ##There is a duplicate metro station
    metro_stations_gdf = metro_stations_gdf.drop_duplicates(subset='name')
    metro_stations_gdf=metro_stations_gdf[~metro_stations_gdf['name'].str.contains('Azər Neft Yağ|Məhəmməd Hadi |B-4',case=False, na=False)].reset_index(drop=True)
    
    return metro_stations_gdf


##Download complete School list in Baku

def download_schools_for_baku():
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json];
    (
      node["amenity"="school"](40.300, 49.700, 40.590, 50.400);
      way["amenity"="school"](40.300, 49.700, 40.590, 50.400);
      relation["amenity"="school"](40.300, 49.700, 40.590, 50.400);
    );
    out body;
    >;
    out skel qt;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    response.raise_for_status()
    data = response.json()
    
    node_coords = {}
    schools = []
    
    # Extract node coordinates
    for element in data['elements']:
        if element['type'] == 'node':
            node_coords[element['id']] = (element['lon'], element['lat'])
            schools.append({
                'id': element['id'],
                'lat': element['lat'],
                'lon': element['lon'],
                'name': element.get('tags', {}).get('name', 'Unnamed'),
                'type': 'node'
            })
    
    # Extract ways and relations
    for element in data['elements']:
        if element['type'] == 'way':
            nodes = [node_coords[node_id] for node_id in element['nodes'] if node_id in node_coords]
            if nodes:
                centroid = Polygon(nodes).centroid
                schools.append({
                    'id': element['id'],
                    'lat': centroid.y,
                    'lon': centroid.x,
                    'name': element.get('tags', {}).get('name', 'Unnamed'),
                    'type': 'way'
                })
        elif element['type'] == 'relation':
            if 'bounds' in element:
                centroid = Polygon([
                    (element['bounds']['minlon'], element['bounds']['minlat']),
                    (element['bounds']['maxlon'], element['bounds']['minlat']),
                    (element['bounds']['maxlon'], element['bounds']['maxlat']),
                    (element['bounds']['minlon'], element['bounds']['maxlat'])
                ]).centroid
                schools.append({
                    'id': element['id'],
                    'lat': centroid.y,
                    'lon': centroid.x,
                    'name': element.get('tags', {}).get('name', 'Unnamed'),
                    'type': 'relation'
                })

    # Create a GeoDataFrame
    geometry = [Point(school['lon'], school['lat']) for school in schools]
    schools_gdf = gpd.GeoDataFrame(schools, geometry=geometry)
    schools_gdf.set_crs("EPSG:4326", inplace=True)
    
    # Drop duplicates based on the 'name' column
    schools_gdf = schools_gdf.drop_duplicates(subset='name')
    schools_gdf= schools_gdf[schools_gdf['name'].str.contains('məktəb|Mәktәb |mekteb|Məktb|okul|Təhsil Kompleksi|Litsey|school|lise|liceum|gimnaziya|kollec|Unnamed', case=False, na=False)]
    values_to_drop = [
    '111 Saylı Məktəbin Həyəti',
    r'183 saylı Məktəb \ Mərdəkan',
    r'246 saylı Məktəb Lisey \ Bakı',
    '252 saylı Məktəb-Lisey',
    r'263 saylı Məktəb \ Günəşli',
    '32 saylı orta məktəb',
    '322 saylı Məktəb',
    '6 saylı Məktəb',
    r'66 saylı məktəb / Tərəqqi liseyi / Montin',
    '70 saylı Məktəb'
]

# Drop rows where 'name' column has the specified values
    schools_gdf = schools_gdf[~schools_gdf['name'].isin(values_to_drop)]
    return schools_gdf


###Download the updated Bus stops data in Baku

def download_parks_for_baku():
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json];
    (
      node["leisure"="park"](40.300, 49.700, 40.590, 50.400);
      way["leisure"="park"](40.300, 49.700, 40.590, 50.400);
      relation["leisure"="park"](40.300, 49.700, 40.590, 50.400);
      
      node["landuse"="forest"](40.300, 49.700, 40.590, 50.400);
      way["landuse"="forest"](40.300, 49.700, 40.590, 50.400);
      relation["landuse"="forest"](40.300, 49.700, 40.590, 50.400);
      
      node["leisure"="garden"](40.300, 49.700, 40.590, 50.400);
      way["leisure"="garden"](40.300, 49.700, 40.590, 50.400);
      relation["leisure"="garden"](40.300, 49.700, 40.590, 50.400);
      
      node["leisure"="nature_reserve"](40.300, 49.700, 40.590, 50.400);
      way["leisure"="nature_reserve"](40.300, 49.700, 40.590, 50.400);
      relation["leisure"="nature_reserve"](40.300, 49.700, 40.590, 50.400);
      
      node["leisure"="recreation_ground"](40.300, 49.700, 40.590, 50.400);
      way["leisure"="recreation_ground"](40.300, 49.700, 40.590, 50.400);
      relation["leisure"="recreation_ground"](40.300, 49.700, 40.590, 50.400);
    );
    out body;
    >;
    out skel qt;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    response.raise_for_status()
    data = response.json()
    
    node_coords = {}
    parks = []
    
    # Extract node coordinates
    for element in data['elements']:
        if element['type'] == 'node':
            node_coords[element['id']] = (element['lon'], element['lat'])
            parks.append({
                'id': element['id'],
                'lat': element['lat'],
                'lon': element['lon'],
                'name': element.get('tags', {}).get('name', 'Unnamed'),
                'type': 'node'
            })
    
    # Extract ways and relations
    for element in data['elements']:
        if element['type'] == 'way':
            nodes = [node_coords[node_id] for node_id in element['nodes'] if node_id in node_coords]
            if nodes:
                centroid = Polygon(nodes).centroid
                parks.append({
                    'id': element['id'],
                    'lat': centroid.y,
                    'lon': centroid.x,
                    'name': element.get('tags', {}).get('name', 'Unnamed'),
                    'type': 'way'
                })
        elif element['type'] == 'relation':
            if 'bounds' in element:
                centroid = Polygon([
                    (element['bounds']['minlon'], element['bounds']['minlat']),
                    (element['bounds']['maxlon'], element['bounds']['minlat']),
                    (element['bounds']['maxlon'], element['bounds']['maxlat']),
                    (element['bounds']['minlon'], element['bounds']['maxlat'])
                ]).centroid
                parks.append({
                    'id': element['id'],
                    'lat': centroid.y,
                    'lon': centroid.x,
                    'name': element.get('tags', {}).get('name', 'Unnamed'),
                    'type': 'relation'
                })

    # Create a GeoDataFrame
    geometry = [Point(park['lon'], park['lat']) for park in parks]
    parks_gdf = gpd.GeoDataFrame(parks, geometry=geometry)
    parks_gdf.set_crs("EPSG:4326", inplace=True)
    
    # Drop duplicates based on the 'name' column
    parks_gdf = parks_gdf.drop_duplicates(subset='name')
    
    return parks_gdf
