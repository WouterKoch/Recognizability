import os.path
# from osgeo import gdal
from tqdm import tqdm
from netCDF4 import Dataset
import numpy as np
import requests, json

cci_codes = [10, 11, 12, 20, 30, 40, 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110, 120, 121, 122, 130, 140, 150,
             151, 152, 153, 160, 170, 180, 190, 200, 201, 202, 210, 220]


def load_file(path):
    file = Dataset(os.path.join(path), 'r')
    array = file.variables['lccs_class']

    array_height = file.dimensions['lat'].size
    array_width = file.dimensions['lon'].size

    return array, array_width, array_height


def get_from_file(lat, lon, array, array_height, array_width):
    lat_pos = int((array_height / 2) - (lat * (array_height / 180)))
    lon_pos = int((array_width / 2) + (lon * (array_width / 360)))
    return int(array[0, lat_pos, lon_pos])

def get_from_api(lat, lon):
    r = requests.post('http://maps.elie.ucl.ac.be/cgi-bin/getCCIlandCover_v207.py', json={"year":"2020","lon":lon,"lat":lat})
    esa = json.loads((r.content).decode("utf-8").replace(")]}',",""))["class"]
    # print(esa)
    return esa

