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


def merge_tif(params, out_put_path_cloud, out_put_path_snow):
    lg = get_logger()
    cloud_tif_path = out_put_path_cloud
    snow_tif_path = out_put_path_snow
    cloudDs = gdal.Open(cloud_tif_path, gdal.GA_ReadOnly)
    snowDs = gdal.Open(snow_tif_path, gdal.GA_ReadOnly)
    if cloudDs is None:
        raise PluginException(1, "临时目录栅格文件不存在", "Required cloud tif")
    elif snowDs is None:
        raise PluginException(1, "临时目录栅格文件不存在", "Required snow tif")
    else:
        pass
    lg.info("Done open tif finshed.")
    geotransform = cloudDs.GetGeoTransform()
    projection = cloudDs.GetProjection()
    cols = cloudDs.RasterXSize
    rows = cloudDs.RasterYSize
    cloudBand = cloudDs.GetRasterBand(1)
    cloudData = cloudDs.ReadAsArray(0,0,cols,rows)
    cloudNoData = cloudBand.GetNoDataValue()
    if cloudNoData is None:
        cloudNoData = 0
    else:
        pass
    if cloudData is None:
        cloudData = 0
    else:
        pass
    snowBand = snowDs.GetRasterBand(1)
    snowData = snowDs.ReadAsArray(0,0,cols,rows)
    snowNoData = snowBand.GetNoDataValue()
    if snowNoData is None:
        snowNoData = 0
    else:
        pass
    if snowData is None:
        snowData = 0
    else:
        pass
    result = snowData
    lg.info("first")
    result = result + cloudData
    
    lg.info("end")
    # write result to disk
    resultPath = params.output_image_path
    
    format = "GTiff" 
    driver = gdal.GetDriverByName(format)
    ds = driver.Create(resultPath, cols, rows, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)
    lg.info("====================")
    # ds.GetRasterBand(1).SetNoDataValue(snowNoData)
    ds.GetRasterBand(1).WriteArray(result)
    ds = None
    lg.info("Done merge tif finshed.")
    # generate browser
    rescale(params.output_image_path, os.path.splitext(params.output_image_path)[0]+"_browser.png", 1024)
    lg.info("Generate browser finshed.")
    # polygonize
    raster_to_polygon(params.output_image_path, params.output_shp_path, 1, True)
    lg.info("Done polygonize.")
    print('Merge tif is all done')