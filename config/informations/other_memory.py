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

import geopandas as gpd
from osgeo import gdal, ogr
import numpy as np
from skimage import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import xml.etree.ElementTree as ET


def area_cal(shpPath):
    '''计算面积'''
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shpPath, 1)
    layer = dataSource.GetLayer()
    area_zong=0
    for feature in layer:
        geom = feature.GetGeometryRef()
        area = geom.GetArea()  # 计算面积
        #area_zong+=area
        # print(area)
        m_area = np.count_nonzero(area)  # 计算像素面积
        area_zong+=m_area
        # layer.SetFeature(feature)
    # dataSource = None
    return layer.GetFeatureCount()


def xml_create(params, area, num, root, image_path):
    ImgFile = ET.SubElement(root,'ImgFile')
    ImageFileName = ET.SubElement(ImgFile,'ImageFileName')
    ImageFileName.text = image_path
    BuildingArea = ET.SubElement(ImgFile,'BuildingArea')
    BuildingArea.text = str(area)
    BuildingNum = ET.SubElement(ImgFile,'BuildingNum')
    BuildingNum.text = str(num)
    tree = ET.ElementTree(root)
    return tree


def pretty_xml(element, indent, newline, level=0):
    if element:
        if (element.text is None) or element.text.isspace():
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)


def merge_shapefiles(shp_files, output_shp):
    # 读取第一个shp文件
    gdf = gpd.read_file(shp_files[0])

    # 逐个读取剩余的shp文件，并将它们与第一个拼接在一起
    for shp_file in shp_files[1:]:
        gdf_temp = gpd.read_file(shp_file)
        gdf = gdf.append(gdf_temp)

    # 重置GeoDataFrame的索引
    gdf.reset_index(drop=True, inplace=True)

    # 将拼接后的shp文件保存到输出路径
    gdf.to_file(output_shp)


def merge_shapefiles_main(shpfiles, output_shp_path):
    shp_files = []
    for filename in os.listdir(shpfiles):
        if 'shp' in filename:
            shp_path= shpfiles + filename
            shp_files.append(shp_path)
    output_shp = output_shp_path
    merge_shapefiles(shp_files, output_shp)