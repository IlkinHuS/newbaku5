import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import h3
import sys
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree
from datetime import date, datetime

# Add your scripts path
sys.path.append('./geo_package')
from geo_package import (
    download_bus_stops_for_baku,
    download_metro_stations_for_baku,
    download_schools_for_baku,
    download_parks_for_baku,
    haversine_distance,
    get_nearest_poi,
    count_poi_within_radius,
    calculate_mean_price_kvm_for_input
)

# For Map
import folium
from streamlit_folium import st_folium

#######################
# Load Data and Models
#######################

import zipfile

@st.cache_resource
def load_data_and_models():
    # Download POI datasets
    bus_stops_gdf = download_bus_stops_for_baku()
    metro_stations_gdf = download_metro_stations_for_baku()
    schools_gdf = download_schools_for_baku()
    parks_gdf = download_parks_for_baku()

    # Main df for mean price calculation
    df = pd.read_csv('df_before_meanprice.csv')
    df['date2'] = pd.to_datetime(df['date2'], errors='coerce')
    df.reset_index(drop=True, inplace=True)

    # Compute radians for BallTree
    df['latitude_rad'] = np.radians(df['latitude'])
    df['longitude_rad'] = np.radians(df['longitude'])
    coords = df[['latitude_rad', 'longitude_rad']].values
    tree = BallTree(coords, metric='haversine')

    # Load compressed XGBoost model from zip
    with zipfile.ZipFile('xgb_target_encoding2.zip', 'r') as zip_ref:
        zip_ref.extractall('temp_model_dir')  # Extract to a temporary directory
    final_model = xgb.Booster()
    final_model.load_model('temp_model_dir/xgb_target_encoding2.json')

    # Load other models and preprocessors
    preprocessor_full = joblib.load('preprocessor_target_encodingjoblib2')
    target_encoder_full = joblib.load('target_encoder2.joblib')

    return bus_stops_gdf, metro_stations_gdf, schools_gdf, parks_gdf, df, tree, final_model, preprocessor_full, target_encoder_full


bus_stops_gdf, metro_stations_gdf, schools_gdf, parks_gdf, df, tree, final_model, preprocessor_full, target_encoder_full = load_data_and_models()










location_cols = ['h3_8']
numerical_cols = [
    'latitude', 'longitude', 'otaq_sayi', 'area_per_room',
    'mean_price_kvm_within_radius7', 'sahe_kvm', 'sahe_square',
    'mertebe_yer', 'mertebe_say', 'mertebe_say_square', 'mertebe_ratio',
    'distance_to_CBD_boundary_km', 'distance_to_nearest_metro', 'distance_to_nearest_bus',
    'distance_to_nearest_school', 'distance_to_nearest_park', 'number_metro_within_800m',
    'number_bus_within_400m', 'number_schl_within_1000m',
    'months_since_start','month_sin','month_cos'
] + location_cols

categorical_cols = ['kateqoriya', 'temir', 'temir_tikili_cat']
all_features = numerical_cols + categorical_cols + ['date2']

#######################
# Streamlit UI
#######################

st.title("Mənzilini qiymətləndir")
#st.write("Enter property details and get the predicted price.")

# Choose Location Input Method
location_method = st.radio(
    "",
    ("Xəritə", "Manual")
)

if location_method == "Manual":
    latitude = st.number_input("Latitude", value=40.3874605, format="%.8f")
    longitude = st.number_input("Longitude", value=49.8030282, format="%.8f")
else:
    st.write("Xəritə üzərindən dəqiq ünvanı seçin")
    # Center map on Baku
    map_center = [40.4093, 49.8671]  # approximate center of Baku
    m = folium.Map(location=map_center, zoom_start=12)
    m.add_child(folium.LatLngPopup())  # Click to get lat/lon popup

    # The returned value will contain information about the last click
    map_data = st_folium(m, width=1200, height=700)
    if map_data and 'last_clicked' in map_data and map_data['last_clicked'] is not None:
        clicked_coords = map_data['last_clicked']
        latitude = clicked_coords['lat']
        longitude = clicked_coords['lng']
        st.success(f"Selected location: Latitude={latitude:.6f}, Longitude={longitude:.6f}")
    else:
        latitude = None
        longitude = None

otaq_sayi = st.number_input("Number of Rooms (otaq sayı)", min_value=1, max_value=10, value=2)
sahe_kvm = st.number_input("Area in sq. meters (sahə-kvm)", min_value=10, max_value=500, value=93)
mertebe_yer = st.number_input("Current Floor (mertebe_yer)", min_value=1, max_value=50, value=10)
mertebe_say = st.number_input("Total Floors (mertebe_say)", min_value=1, max_value=50, value=19)
date2_input = st.date_input("Date (date2)", value=date(2024,12,14))

kateqoriya = st.selectbox("Category (kateqoriya)", ['Köhnə tikili', 'Yeni tikili'])
temir = st.selectbox("Renovation (temir)", ['var', 'yoxdur'])

# Calculate temir_tikili_cat based on kateqoriya and temir
temir_tikili_cat = f"{kateqoriya}_{temir}"

predict_button = st.button("Predict Price")

if predict_button:
    if location_method == "Select from Map" and (latitude is None or longitude is None):
        st.error("Please select a location on the map first.")
    else:
        # Build input_data dictionary
        input_data = {
            'latitude': float(latitude),
            'longitude': float(longitude),
            'otaq_sayi': float(otaq_sayi),
            'sahe_kvm': float(sahe_kvm),
            'mertebe_yer': float(mertebe_yer),
            'mertebe_say': float(mertebe_say),
            'date2': date2_input.strftime('%Y-%m-%d'),
            'kateqoriya': kateqoriya,
            'temir': temir,
            'temir_tikili_cat': temir_tikili_cat
        }

        # Prepare input_df
        input_df = pd.DataFrame([input_data])

        # Compute h3_8
        input_df['h3_8'] = input_df.apply(lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], 8),axis=1)

        # Calculate area_per_room
        input_df['area_per_room'] = input_df['sahe_kvm'] / input_df['otaq_sayi']

        # Calculate mean_price_kvm_within_radius7
        mean_price_val = calculate_mean_price_kvm_for_input(df, input_data, tree=tree)
        input_df['mean_price_kvm_within_radius7'] = mean_price_val

        # Additional features
        input_df['sahe_square']= np.square(input_df['sahe_kvm'])
        input_df['mertebe_say_square'] = np.square(input_df['mertebe_say'])
        input_df['mertebe_ratio'] = input_df['mertebe_yer'] / input_df['mertebe_say']

        # Distance to CBD boundary
        cbd_lat, cbd_lon = 40.3709087, 49.8361090
        distance_to_CBD_km = haversine_distance(latitude, longitude, cbd_lat, cbd_lon)
        radius_km = 0.5
        distance_to_CBD_boundary_km = max(0, distance_to_CBD_km - radius_km)
        input_df['distance_to_CBD_boundary_km'] = distance_to_CBD_boundary_km

        # House coords in radians
        house_coords = np.radians([[latitude, longitude]])

        # Nearest POIs
        input_df = get_nearest_poi(input_df, metro_stations_gdf, house_coords, 'metro', k_neighbors=1)
        input_df = get_nearest_poi(input_df, bus_stops_gdf, house_coords, 'bus', k_neighbors=1)
        input_df = get_nearest_poi(input_df, schools_gdf, house_coords, 'school', k_neighbors=1)
        input_df = get_nearest_poi(input_df, parks_gdf, house_coords, 'park', k_neighbors=1)

        # Count POI within certain radii
        geometry = [Point(longitude, latitude)]
        data_format_gdf = gpd.GeoDataFrame(input_df, geometry=geometry)
        data_format_gdf = data_format_gdf.set_crs("EPSG:4326")
        buffer_distances = [400, 800, 1200]
        input_df = count_poi_within_radius(input_df, data_format_gdf, metro_stations_gdf, buffer_distances, 'metro')
        input_df = count_poi_within_radius(input_df, data_format_gdf, bus_stops_gdf, buffer_distances, 'bus')

        all_cols = numerical_cols + categorical_cols + ['date2']
        # Drop columns not in all_cols
        cols_to_drop = [col for col in input_df.columns if col not in all_cols]
        input_df = input_df.drop(cols_to_drop, axis=1)

        # Count number of schools within 1 km
        buffer_distances_sch = [1000]
        input_df = count_poi_within_radius(input_df, data_format_gdf, schools_gdf, buffer_distances_sch, 'schl')

        # Calculate months_since_start
        start_date = df['date2'].min()
        input_df['date2'] = pd.to_datetime(input_df['date2'])
        input_df['months_since_start'] = ((input_df['date2'] - start_date).dt.days / 30.44).round(0)
        input_df['months_since_start'] = input_df['months_since_start'].astype('Int64')

        # Calculate month_sin and month_cos
        input_df['month'] = input_df['date2'].dt.month
        input_df['month_sin'] = np.sin(2 * np.pi * input_df['month'] / 12)
        input_df['month_cos'] = np.cos(2 * np.pi * input_df['month'] / 12)

        # Round latitude and longitude
        input_df['latitude'] = input_df['latitude'].round(3)
        input_df['longitude'] = input_df['longitude'].round(3)

        # Handle missing cols
        missing_cols = [col for col in (numerical_cols+categorical_cols) if col not in input_df.columns]
        for col in missing_cols:
            input_df[col] = np.nan

        # Ensure column order
        input_df = input_df[numerical_cols + categorical_cols]

        # Transform 'h3_8'
        input_df[location_cols] = target_encoder_full.transform(input_df[location_cols])

        # Preprocess
        input_processed = preprocessor_full.transform(input_df)

        # Feature names
        cat_feature_names_full = preprocessor_full.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
        feature_names_full = np.concatenate([numerical_cols, cat_feature_names_full]).tolist()

        # Predict
        dinput = xgb.DMatrix(input_processed, feature_names=feature_names_full)
        prediction = final_model.predict(dinput)
        predicted_price = np.exp(prediction)[0]  # Since price was log-transformed
        predicted_price_per_sqm = predicted_price / input_df['sahe_kvm'].values[0]
        st.success(f"Predicted Price: {predicted_price:.2f}, Predicted Price per sqm: {predicted_price_per_sqm:.2f}, Mean price within 500 meter: {mean_price_val:.2f}")
        #st.success(f"Predicted Price: {predicted_price:.2f}, Predicted Price per sqm: {predicted_price_per_sqm:.2f}")



