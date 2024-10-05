
import requests
import json

def fetch_from_nasa_api(api_url, params):
    """
    Fetch data from NASA API with given parameters.
    """
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return None

def convert_geotiff_to_array(filepath):
    """
    Convert GeoTIFF file to a NumPy array for further processing.
    """
    with rasterio.open(filepath) as src:
        array = src.read(1)
    return array
