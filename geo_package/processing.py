###Find distance to CBD
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from math import radians, cos, sin, sqrt, atan2
from sklearn.neighbors import BallTree
from datetime import datetime
from dateutil.relativedelta import relativedelta

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c


##Create a function to find the distance to the nearest object (school, metro, park etc.,)

def get_nearest_poi(df, poi_gdf, house_coords, poi_name, k_neighbors=1):
    """
    Calculate the distance to the nearest point of interest (POI) for each house.
    
    Parameters:
    - df (GeoDataFrame): The DataFrame containing house locations to be updated.
    - house_coords (numpy array): Array of house coordinates in radians.
    - poi_gdf (GeoDataFrame): GeoDataFrame containing POI locations.
    - poi_name (str): Name of the point of interest (for column naming).
    - k_neighbors (int): Number of nearest neighbors to find. Default is 1.
    
    Returns:
    - GeoDataFrame: Updated df with the distance to the nearest POI.
    """
    # Extract POI coordinates and convert to radians
    poi_coords = np.radians(poi_gdf[['lat', 'lon']])
    
    # Build a BallTree for fast nearest-neighbor lookup
    tree = BallTree(poi_coords, metric='haversine')
    
    # Query the BallTree to find the distance to the nearest POI
    distances, indices = tree.query(house_coords, k=k_neighbors)
    
    # Convert distances from radians to meters (Earth's radius = 6371000 meters)
    distances_meters = distances * 6371000
    
    # Add the distance to the nearest POI to the house DataFrame
    column_name = f'distance_to_nearest_{poi_name}'
    df[column_name] = distances_meters.flatten()
    
    return df


####Create a function to calculate number of POI within 400, 800, 1200 meters
def count_poi_within_radius(df, houses_gdf, poi_gdf, distances, stop_type):
    """
    Count the number of poi (e.g., metro, bus) within specified distances around each house.
    
    Parameters:
    houses_gdf (GeoDataFrame): GeoDataFrame containing house locations.
    poi_gdf (GeoDataFrame): GeoDataFrame containing stop locations (e.g., metro, bus).
    distances (list): List of distances (in meters) to create buffers around houses.
    stop_type (str): Type of poi (e.g., 'metro', 'bus') for column naming.

    Returns:
    DataFrame: Updated DataFrame with counts of stops within specified distances.
    """
    # Reproject to a projected CRS for accurate distance buffering
    houses_gdf2 = houses_gdf.to_crs(epsg=32639)  # UTM zone 39N for Baku
    poi_gdf2 = poi_gdf.to_crs(epsg=32639)
    
    # Ensure the CRS match
    assert houses_gdf2.crs == poi_gdf2.crs, "CRS do not match!"
    
    for distance in distances:
        # Create buffer around each house
        buffer_column = f'buffer_{distance}m_{stop_type}'
        houses_gdf2[buffer_column] = houses_gdf2.geometry.buffer(distance)
        
        # Set the buffer column as the active geometry
        houses_gdf2 = houses_gdf2.set_geometry(buffer_column)
        
        # Spatial join to count the number of poi within the buffer
        poi_within_buffer = gpd.sjoin(houses_gdf2, poi_gdf2, how='left', predicate='contains')
        poi_within_buffer = poi_within_buffer.dropna(subset=['index_right'])
        
        # Count the number of poi within each buffer
        poi_counts = poi_within_buffer.groupby(poi_within_buffer.index).size()
        
        # Add the counts back to the original DataFrame
        count_column = f'number_{stop_type}_within_{distance}m'
        df[count_column] = poi_counts.reindex(houses_gdf.index).fillna(0).astype(int)
        
        # Drop the buffer column and reset geometry
        houses_gdf2 = houses_gdf2.drop(columns=buffer_column)
        houses_gdf2 = houses_gdf2.set_geometry('geometry')
    
    return df

###!!requires date in pd.datetime format!!!
def calculate_mean_price_kvm_for_input(df, input_data, tree, radius_meters=500, months=3, sigma=140):
    """
    Calculate the Gaussian-weighted mean price per square meter (qiymet_kvm)
    for a single input observation based on neighbors in df.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame must contain:
        - 'latitude', 'longitude' (floats)
        - 'date2' (datetime)
        - 'qiymet_kvm' (float)
        - 'kateqoriya' (str)
        - 'sahe_kvm' (float) [if needed for future filters]
    input_data : dict
        Dictionary representing a single observation with keys:
        - 'latitude', 'longitude'
        - 'date2' (parseable as datetime)
        - 'kateqoriya'
        - 'sahe_kvm'
    radius_meters : float, optional
        Radius in meters to consider for neighbors. Default = 500m.
    months : int, optional
        Lookback period in months. Default = 3.
    sigma : float, optional
        Bandwidth for Gaussian kernel in meters. Default = 140.

    Returns
    -------
    input_data : dict
        The input_data dictionary updated with a new key:
        'mean_price_kvm_within_radius7' containing the computed value (float or NaN).
    """

    # Ensure date2 is datetime in df and input_data
    input_date = pd.to_datetime(input_data['date2'])

    # Convert input lat/long to radians
    input_lat_rad = np.radians(input_data['latitude'])
    input_lon_rad = np.radians(input_data['longitude'])

    # Prepare df coords in radians if not already done
    if 'latitude_rad' not in df.columns or 'longitude_rad' not in df.columns:
        df['latitude_rad'] = np.radians(df['latitude'])
        df['longitude_rad'] = np.radians(df['longitude'])

    # Earth's radius in meters
    earth_radius = 6371000
    # Convert radius to radians
    radius = radius_meters / earth_radius

    # Build BallTree

    # Query neighbors within radius
    neighbors_indices, neighbors_distances = tree.query_radius(
        [[input_lat_rad, input_lon_rad]], r=radius, return_distance=True
    )

    # Unpack the single-element lists
    neighbors_indices = neighbors_indices[0]
    neighbors_distances = neighbors_distances[0]

    # Convert distances from radians to meters
    neighbors_distances = neighbors_distances * earth_radius

    # Exclude the hypothetical "self" match scenario by default
    # (If the input data is not in df, this isn't needed, but just in case)
    # No direct index for input_data in df, so no need for self-exclusion here

    if len(neighbors_indices) == 0:
        input_data['mean_price_kvm_within_radius7'] = np.nan
        return input_data

    # Date range: input_date - months to input_date
    start_date = input_date - pd.DateOffset(months=months)
    end_date = input_date

    # Filter neighbors in df by category and date range
    filtered_df = df.iloc[neighbors_indices].copy()
    filtered_df = filtered_df[
        (filtered_df['date2'] >= start_date) &
        (filtered_df['date2'] <= end_date) &
        (filtered_df['kateqoriya'] == input_data['kateqoriya'])
        # Add more filters if desired, e.g. by size_category
    ]

    if filtered_df.empty:
        input_data['mean_price_kvm_within_radius7'] = np.nan
        return input_data

    # Map filtered indices back to neighbors_distances
    filtered_indices = filtered_df.index.values
    intersect_indices, idx_neighbors, idx_filtered = np.intersect1d(
        neighbors_indices, filtered_indices, return_indices=True
    )

    if len(intersect_indices) == 0:
        input_data['mean_price_kvm_within_radius7'] = np.nan
        return input_data

    mapped_distances = neighbors_distances[idx_neighbors]

    epsilon = 1e-5  # Avoid division by zero
    mapped_distances = np.maximum(mapped_distances, epsilon)

    # Compute Gaussian weights
    weights = np.exp(-0.5 * (mapped_distances / sigma)**2)
    weights /= weights.sum()

    filtered_prices = filtered_df['qiymet_kvm'].iloc[idx_filtered].values
    weighted_mean_price_kvm = np.dot(filtered_prices, weights)

    #input_data['mean_price_kvm_within_radius7'] = weighted_mean_price_kvm

    return weighted_mean_price_kvm
   


