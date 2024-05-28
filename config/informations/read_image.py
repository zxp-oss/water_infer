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
import tqdm


def rescale_image(params, resrange, image_path, image_name):
    lg = get_logger()
    resrange = resrange
    res = get_resolution(image_path)
    best_res = resrange[0]
    if res is not None:
        if res <= resrange[0]:
            image_path, _, _, _, _ = rescale(image_path, os.path.join(params.work_dir, "rescale" + image_name + ".tif"), factor=res/best_res)
        elif resrange[0] <= res <= resrange[1]:
            pass
        else:
            raise PluginException(1, "不支持的影像分辨率：%s，当前仅支持%s-%sm分辨率影像" % (res, resrange[0], resrange[1]) , "Required image res level is %s - %s m, got %s" % (res, resrange[0], resrange[1]))
    else:
        lg.log(logging.INFO, "Image is not have res")
        pass

    return image_path, res


def read_image(params):
    lg = get_logger()
    image_path = params.input_image_path
    image_name = os.path.basename(image_path).rsplit('.', 1)[0]
    # verify image pixel type and number of bands
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    width = ds.RasterXSize
    height = ds.RasterYSize
    nband = ds.RasterCount
    if nband == 1:
        raise PluginException(1, "不支持的影像通道类型：%s，输入影像通道数必须为%s波段或%s波段" % (nband, 3, 4),
                              "Unsupported nband type: %s, has to be %s nband or %s nband" % (nband, 3, 4))
    else:
        pass
    dtype = ds.GetRasterBand(1).DataType
    if dtype != gdal.GDT_Byte and dtype != gdal.GDT_UInt16:
        ds = None
        dtype_name = gdal.GetDataTypeName(dtype)
        uint8_name = gdal.GetDataTypeName(gdal.GDT_Byte)
        uint16_name = gdal.GetDataTypeName(gdal.GDT_UInt16)
        raise PluginException(1, "不支持的像素类型：%s，输入影像必须为%s或%s" % (dtype_name, uint8_name, uint16_name),
                              "Unsupported pixel type: %s, has to be %s or %s" % (dtype_name, uint8_name, uint16_name))
    ds = None
    resrange = [params.rangemin, params.rangemax]
    image_path, res = rescale_image(params, resrange, image_path, image_name)

    return image_path, res


def get_block_coords(height, width, block_size, block_stride):
    coords = []
    for x in range(0, width, block_stride):
        if x + block_size > width:
            block_width = width - x
        else:
            block_width = block_size
        for y in range(0, height, block_stride):
            if y + block_size > height:
                block_height = height - y
            else:
                block_height = block_size
            coords.append((x, y, block_width, block_height))
    return coords


def read_image_bfp(params):
    lg = get_logger()
    image_path = params.input_image_path
    image_name = os.path.basename(image_path).rsplit('.', 1)[0]
    # verify image pixel type and number of bands
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    width = ds.RasterXSize
    height = ds.RasterYSize
    nband = ds.RasterCount
    dtype = ds.GetRasterBand(1).DataType

    if dtype != gdal.GDT_Byte and dtype != gdal.GDT_UInt16:
        ds = None
        dtype_name = gdal.GetDataTypeName(dtype)
        uint8_name = gdal.GetDataTypeName(gdal.GDT_Byte)
        uint16_name = gdal.GetDataTypeName(gdal.GDT_UInt16)
        raise PluginException(1, "不支持的像素类型：%s，输入影像必须为%s或%s" % (dtype_name, uint8_name, uint16_name),
                              "Unsupported pixel type: %s, has to be %s or %s" % (dtype_name, uint8_name, uint16_name))
    ds = None
    resrange = [params.rangemin, params.rangemax]
    image_path, res = rescale_image(params, resrange, image_path, image_name)

    return image_path, nband, res


def rescal_image_back(params, path1, path2, path3):
    resrange = [params.rangemin, params.rangemax]
    res = get_resolution(params.input_image_path)
    best_res = resrange[0]
    path2 = os.path.join(os.path.dirname(path1), os.path.basename(path1).rsplit('.', 1)[0] + "_ori" + ".tif")
    rescale_back(path1, path2, path3)
    print('best res is', best_res)
    return res, path2


def rescale_back(srcpath, dstpath, oripath):
    ds = gdal.Open(srcpath, gdal.GA_ReadOnly)
    ncol = ds.RasterXSize
    nrow = ds.RasterYSize
    nband = ds.RasterCount
    print("Output temp image row, col, band: (%d, %d, %d)" % (nrow, ncol, nband))
    
    ds1 = gdal.Open(oripath, gdal.GA_ReadOnly)
    new_ncol = ds1.RasterXSize
    new_nrow = ds1.RasterYSize
    new_nband = ds1.RasterCount
    print("Input image row, col, band: (%d, %d, %d)" % (new_nrow, new_ncol, new_nband))
    
    gdal.Translate(dstpath, ds, width=new_ncol, height=new_nrow, resampleAlg=gdal.GRA_NearestNeighbour)
    print("Rescaledback shape: (%d, %d, %d)" % (new_nrow, new_ncol, nband))
    ds = None
    ds1 = None
    
    return dstpath, nrow, ncol, new_nrow, new_ncol