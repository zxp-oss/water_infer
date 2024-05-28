import os
import argparse
import logging
import traceback
import os.path as osp
import sys
def get_root():
    rdir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    return rdir
sys.path.append(osp.join(get_root(), 'config'))
from config.utils import get_rootdir, get_logger, crop_extent, PluginException, \
    linear_stretch, stretch, Fishnet, assign_spatial_reference_byfile, get_resolution, rescale,\
    raster_to_polygon
from osgeo import gdal, gdalconst
import numpy as np
from skimage import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


# 居民地拉伸卫星类型
def stretch_all_you_need_lbp(params, image_name, block_image):
    lg = get_logger()
    if block_image.dtype == np.uint16:
        block_image = stretch(block_image)
    elif image_name[0:2].upper() == 'GF':
        block_image = stretch(block_image)
    elif image_name[0:2].upper() == 'TH':
        if image_name[0:4].upper() == 'TH01':
            if image_name[0:4].upper() == 'TH01':
                #block_image = stretch(block_image)
                pass
            elif image_name[0:7].upper() == 'TH01-01':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH01-02':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH01-03':
                pass
                #block_image = stretch(block_image)
            else:
                pass
        elif image_name[0:4].upper() == 'TH02':
            if image_name[0:4].upper() == 'TH02':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH02-01':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH02-02':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH02-03':
                pass
                #block_image = stretch(block_image)
            else:
                pass
        elif image_name[0:4].upper() == 'TH03':
            if image_name[0:4].upper() == 'TH01':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH03-01':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH03-02':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH03-03':
                block_image = stretch(block_image)
            else:
                pass
    else:
        pass
    
    lg.log(logging.INFO, "Stretch is all lbp need.")
    return block_image


# 水体拉伸卫星类型
def stretch_all_you_need_water(params, image_name, block_image):
    lg = get_logger()
    if block_image.dtype == np.uint16:
        block_image = stretch(block_image)
    elif image_name[0:2].upper() == 'GF':
        block_image = stretch(block_image)
    elif image_name[0:2].upper() == 'TH':
        if image_name[0:4].upper() == 'TH01':
            if image_name[0:4].upper() == 'TH01':
                #block_image = stretch(block_image)
                pass
            elif image_name[0:7].upper() == 'TH01-01':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH01-02':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH01-03':
                pass
                #block_image = stretch(block_image)
            else:
                pass
        elif image_name[0:4].upper() == 'TH02':
            if image_name[0:4].upper() == 'TH02':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH02-01':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH02-02':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH02-03':
                pass
                #block_image = stretch(block_image)
            else:
                pass
        elif image_name[0:4].upper() == 'TH03':
            if image_name[0:4].upper() == 'TH01':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH03-01':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH03-02':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH03-03':
                block_image = stretch(block_image)
            else:
                pass
    else:
        pass
    
    lg.log(logging.INFO, "Stretch is all water need.")
    return block_image


# 植被卫星拉伸类型整合
def stretch_all_you_need_vegetation(params, image_name, block_image):
    lg = get_logger()
    if block_image.dtype == np.uint16:
        block_image = stretch(block_image)
    elif image_name[0:2].upper() == 'GF':
        block_image = stretch(block_image)
    elif image_name[0:2].upper() == 'TH':
        if image_name[0:4].upper() == 'TH01':
            if image_name[0:4].upper() == 'TH01':
                #block_image = stretch(block_image)
                pass
            elif image_name[0:7].upper() == 'TH01-01':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH01-02':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH01-03':
                pass
                #block_image = stretch(block_image)
            else:
                pass
        elif image_name[0:4].upper() == 'TH02':
            if image_name[0:4].upper() == 'TH02':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH02-01':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH02-02':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH02-03':
                pass
                #block_image = stretch(block_image)
            else:
                pass
        elif image_name[0:4].upper() == 'TH03':
            if image_name[0:4].upper() == 'TH01':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH03-01':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH03-02':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH03-03':
                block_image = stretch(block_image)
            else:
                pass
    else:
        pass
    
    lg.log(logging.INFO, "Stretch is all vegetation need.")
    return block_image


# 云检测卫星类型拉伸整合
def stretch_all_you_need_cloud(params, image_name, block_image):
    lg = get_logger()
    if block_image.dtype == np.uint16:
        block_image = stretch(block_image)
    elif image_name[0:2].upper() == 'GF':
        block_image = stretch(block_image)
    elif image_name[0:2].upper() == 'TH':
        if image_name[0:4].upper() == 'TH01':
            if image_name[0:4].upper() == 'TH01':
                #block_image = stretch(block_image)
                pass
            elif image_name[0:7].upper() == 'TH01-01':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH01-02':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH01-03':
                pass
                #block_image = stretch(block_image)
            else:
                pass
        elif image_name[0:4].upper() == 'TH02':
            if image_name[0:4].upper() == 'TH02':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH02-01':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH02-02':
                pass
                #block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH02-03':
                pass
                #block_image = stretch(block_image)
            else:
                pass
        elif image_name[0:4].upper() == 'TH03':
            if image_name[0:4].upper() == 'TH01':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH03-01':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH03-02':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH03-03':
                block_image = stretch(block_image)
            else:
                pass
    else:
        pass
    
    lg.log(logging.INFO, "Stretch is all cloud need.")
    return block_image


# 雪检测卫星拉伸类型整合
def stretch_all_you_need_snow(params, image_name, block_image):
    lg = get_logger()
    if block_image.dtype == np.uint16:
        block_image = stretch(block_image)
    elif image_name[0:2].upper() == 'GF':
        block_image = stretch(block_image)
    elif image_name[0:2].upper() == 'TH':
        if image_name[0:4].upper() == 'TH01':
            if image_name[0:4].upper() == 'TH01':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH01-01':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH01-02':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH01-03':
                block_image = stretch(block_image)
            else:
                pass
        elif image_name[0:4].upper() == 'TH02':
            if image_name[0:4].upper() == 'TH02':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH02-01':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH02-02':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH02-03':
                block_image = stretch(block_image)
            else:
                pass
        elif image_name[0:4].upper() == 'TH03':
            if image_name[0:4].upper() == 'TH01':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH03-01':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH03-02':
                block_image = stretch(block_image)
            elif image_name[0:7].upper() == 'TH03-03':
                block_image = stretch(block_image)
            else:
                pass
    else:
        pass
    
    lg.log(logging.INFO, "Stretch is all snow need.")
    return block_image