import os
import ctypes
import argparse
import logging
import traceback
from config.category.process_block_water import *
from process_memory import process_main
from osgeo import gdal, gdalconst
import numpy as np
from skimage import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import tqdm
import torch
from torchvision import transforms

def main():
    lg = get_logger()
    parser = argparse.ArgumentParser()
    # parser.add_argument("license_path")
    parser.add_argument("input_json")
    # parser.add_argument("idlist")
    args = parser.parse_args()

    # Get parameters from job_order.json
    params = ParamsParser(args.input_json)  # parse params in json file
    params.parse()
    #so = ctypes.CDLL("../../InterPlugin/GEOVISDIR/lib/libGeovisLicense.so")
    #volid_ctypres = so.GeovisLicenseRegister()
    #if volid_ctypres == 1:
        #raise PluginException(1, "加载许可驱动失败", "Failed to load the license driver")
        #break
    #else:
        #print("version_2.0")
        #pass
    lg.log(logging.INFO, "Done parsing parameters.")

    params.write_progress(0)
    process_main(params)
    params.write_progress(100)
    lg.log(logging.INFO, "All finished!")
    return params