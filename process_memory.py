import os
import ctypes
import argparse
import logging
import traceback
from config.category.process_block_water import *
from osgeo import gdal, gdalconst

from main import *
import numpy as np
from skimage import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import tqdm
import torch
from torchvision import transforms


def process_main(params):
    lg = get_logger()
    # cloud inference
    if params.target_name == 'cloud':
        out_put_path = params.output_image_path
        out_put_shp_path = params.output_shp_path
        model_path = params.weights
        progess_addnum = 40
        progess_basenum = 10
        process_cloud(params, model_path, out_put_path, params.target_name, progess_addnum, progess_basenum, out_put_shp_path)
    # snow inference
    elif params.target_name == 'snow':
        out_put_path = params.output_image_path
        out_put_shp_path = params.output_shp_path
        model_path = params.weights
        progess_addnum = 40
        progess_basenum = 10
        process_snow(params, model_path, out_put_path, params.target_name, progess_addnum, progess_basenum, out_put_shp_path)
    # cloud and snow inference
    elif params.target_name == 'SnowCloud':
        # cloud inference
        params.target_name == 'cloud'
        model_path_cloud = params.weights_cloud
        out_put_path_cloud = os.path.join(params.work_dir + 'cloud' + '.tif')
        out_put_shp_path = os.path.join(params.work_dir + 'cloud' + '.shp')
        progess_addnum = 0
        progess_basenum = 10
        process_cloud(params, model_path_cloud, out_put_path_cloud, params.target_name, progess_addnum, progess_basenum, out_put_shp_path)
        
        # snow inference
        params.target_name == 'snow'
        out_put_path_snow = os.path.join(params.work_dir + 'snow' + '.tif')
        out_put_shp_path = os.path.join(params.work_dir + 'snow' + '.shp')
        model_path_snow = params.weights_snow
        progess_addnum = 0
        progess_basenum = 50
        process_snow(params, model_path_snow, out_put_path_snow, params.target_name, progess_addnum, progess_basenum, out_put_shp_path)
        
        # merge tif
        merge_tif(params, out_put_path_cloud, out_put_path_snow)
    # lbp inference
    elif params.target_name == 'lbp':
        # out_put_path = params.output_image_path
        # out_put_shp_path = params.output_shp_path
        # model_path = params.weights
        # progess_addnum = 40
        # progess_basenum = 10
        process_lbp(params)
    # vegetation inference
    elif params.target_name == 'vegetation':
        # out_put_path = params.output_image_path
        # out_put_shp_path = params.output_shp_path
        # model_path = params.weights
        # progess_addnum = 40
        # progess_basenum = 10
        process_vegetation(params)
    # water inference
    elif params.target_name == 'water':
        # out_put_path = params.output_image_path
        # out_put_shp_path = params.output_shp_path
        # model_path = params.weights
        # progess_addnum = 40
        # progess_basenum = 10
        process_water(params)
    # bfp inference
    elif params.target_name == 'bfp':
        # out_put_path = params.output_image_path
        # out_put_shp_path = params.output_shp_path
        # model_path = params.weights
        # progess_addnum = 40
        # progess_basenum = 10
        process_bfp(params)
        
    else:
        pass

    lg.info("Done processing.")

if __name__ == '__main__':

    ret_code = 0
    ret_analyse_eng = "SUCCESS"
    ret_analyse_chn = "运行成功"
    params = None
    lg = get_logger()

    try:
        params = main()
    except Exception as e:
        ret_code = getattr(e, "ret_code", -1)
        ret_analyse_chn = getattr(e, "ret_analyse_chn", "运行错误")
        ret_analyse_eng = getattr(e, "ret_analyse_eng", "Program returned an error.")
        lg.log(logging.ERROR, str(e))
        traceback.print_exc()
    finally:
        try:
            if params is not None:
                params.set_return_info(ret_code, ret_analyse_chn, ret_analyse_eng)
                params.dump()
        except Exception as e:
            lg.log(logging.ERROR, "Failed to dump params")
            lg.log(logging.ERROR, str(e))