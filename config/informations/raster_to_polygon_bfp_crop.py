import cv2
import os
import sys
import logging
import json
import uuid
from io import BytesIO
from PIL import Image
from skimage import io, img_as_ubyte
import numpy as np
from pyproj import CRS, Transformer
from osgeo import gdal, ogr, osr
import math
os.environ['RESTAPI_USE_ARCPY'] = 'FALSE'
import requests
import math
import shutil
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union,polygonize
import shapely
from shapely.validation import explain_validity
import os.path as osp
import sys
def get_root():
    rdir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    return rdir
sys.path.append(osp.join(get_root(), 'config'))
from config.utils import get_rootdir, get_logger, crop_extent, PluginException, \
    linear_stretch, stretch, Fishnet, assign_spatial_reference_byfile, get_resolution, rescale,\
    raster_to_polygon
from config.ops.boundary_optimization_crop import process_boundary



def validate_geometry(geom):
    if not geom.is_valid:
        geom = geom.buffer(0)

    if not geom.is_valid:
        points = MultiPoint(polygon.exterior.coords[1:])
        geom = points.convex_hull

    return geom


def merge_vectors_crop(src_dir, dst_path):

    VECTOR_DRIVER = {
        "shp": "ESRI Shapefile",
        "json": "GeoJSON",
        "geojson": "GeoJSON"
    }
    src_vector_files = []
    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)
        src_ext = os.path.splitext(src_path)[1]
        if src_ext in ['.shp', '.json', '.geojson']:
            src_vector_files.append(src_path)
    if len(src_vector_files) == 0:
        return None
    src_ext = os.path.splitext(src_vector_files[0])[1]
    src_drv = ogr.GetDriverByName(VECTOR_DRIVER.get(src_ext[1:]))
    src_ds = src_drv.Open(src_vector_files[0])
    src_layer = src_ds.GetLayer()
    crs = src_layer.GetSpatialRef()
    if crs is None:
        crs = ''
    else:
        crs = crs.ExportToWkt()
    src_ds.Destroy()
    concat_data = pd.concat([gpd.read_file(src_path) for src_path in src_vector_files]).pipe(gpd.GeoDataFrame)
    geoms = concat_data.geometry.tolist()
    geoms = [validate_geometry(g) for g in geoms]
    geoms = unary_union(geoms)
    union_data = gpd.GeoSeries(geoms, crs=crs).explode()
    geoms = [validate_geometry(g) for g in union_data.geometry.tolist()]
    dst_data = gpd.GeoSeries(geoms, crs=crs)
    dst_data.to_file(dst_path, crs=crs, encoding='utf-8')
    return dst_path


def raster_to_polygon_bfp_crop(src, dst, resolution=0.5, xoff=None, yoff=None, xsize=None, ysize=None):

    VECTOR_DRIVER = {
        "shp": "ESRI Shapefile",
        "json": "GeoJSON",
        "geojson": "GeoJSON"
    }
    process_boundary(src, dst, 'bfp', resolution, xoff, yoff, xsize, ysize)

    dst_lyrname, dst_ext = os.path.splitext(os.path.basename(dst))
    dst_drvname = VECTOR_DRIVER.get(dst_ext[1:])
    drv = ogr.GetDriverByName(dst_drvname)

    ds = drv.Open(dst, update=1)
    lyr = ds.GetLayer()
    
    error_inxs = []
    for i in range(lyr.GetFeatureCount()):
        fea = lyr.GetFeature(i)
        geom = fea.GetGeometryRef()
        
        try:
            geom = shapely.wkt.loads(geom.ExportToWkt())
            
            geom = validate_geometry(geom)

            fea.SetGeometry(ogr.CreateGeometryFromWkt(shapely.wkt.dumps(geom)))

            fea = None
            
        except Exception as e:
            print("shapely load from wkt failed: ", str(e))
            #print("==========", geom.ExportToWkt())
            error_inxs.append(i)
    
    
    for inx in error_inxs:
        fea = lyr.GetFeature(inx)
        lyr.DeleteFeature(fea.GetFID())
        fea = None
    lyr = None
    ds = None

    return dst


def get_patch_coords(height, width, patch_size, patch_stride):
    coords = []
    for x in range(0, width, patch_stride):
        if x + patch_size > width:
            patch_width = width - x
        else:
            patch_width = patch_size
        for y in range(0, height, patch_stride):
            if y + patch_size > height:
                patch_height = height - y
            else:
                patch_height = patch_size
            coords.append((x, y, patch_width, patch_height))
    return coords


def raster_to_polygon_bfp_crop_main(params, img_height, img_width, res):
    # polygonize
    lg = get_logger()
    polygon_dir = os.path.join(params.work_dir, 'raster_to_polygon_bfp_crop')
    os.makedirs(polygon_dir, exist_ok=True)
    coords = get_patch_coords(img_height, img_width, 10000, 10000)

    for i in range(len(coords)):
        x, y, w, h = coords[i]
        raster_to_polygon_bfp_crop(params.output_image_path, os.path.join(polygon_dir, f'{x}_{y}_{w}_{h}.shp'), res, x, y, w, h)

    merged_vector_file = merge_vectors_crop(polygon_dir, params.output_shp_path)
    lg.info("Done crop polygonize.")