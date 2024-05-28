#include <vector>
# -*- coding:utf-8 -*- 
import os
import sys
import logging
import json
from skimage import io, img_as_ubyte
import numpy as np
from pyproj import CRS, Transformer
from osgeo import gdal, ogr, osr
import math
import os.path as osp
def get_root():
    rdir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    return rdir
sys.path.append(osp.join(get_root(), 'config'))
from config.ops.boundary_optimization import process_boundary
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
import shapely


def get_rootdir():
    rdir = None
    if getattr(sys, 'frozen', False):
        rdir = os.path.dirname(os.path.abspath(sys.executable))
    elif __file__:
        rdir = os.path.dirname(os.path.abspath(__file__))
    return rdir


def get_logger(lg_name='my_logger', write_to_file=False):
    lg = logging.Logger(name = lg_name)
    if lg.hasHandlers():
        pass
    else:
        formatter_attr = '[%(asctime)s %(levelname)s][%(filename)s:%(lineno)d] %(message)s'
        formatter = logging.Formatter(formatter_attr)

        if write_to_file:
            global_log_dir = os.path.dirname(os.path.abspath(__file__))
            fh = logging.FileHandler(os.path.join(global_log_dir, 'process.log'))
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            lg.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(logging.DEBUG)
        lg.addHandler(ch)

    return lg


def linear_stretch(img):
    minv = np.min(img)
    maxv = np.max(img)
    img_nor = (img - minv) / (maxv - minv)
    return img_nor


def stretch(img):
    n = img.shape[2]
    for i in range(n):
        c1 = img[:, :, i]
        c = np.percentile(c1[c1>0], 2)  # 只拉伸大于零的值
        d = np.percentile(c1[c1>0], 98)
        t = 255 * (img[:, :, i] - c) / (d - c)
        t[t < 0] = 0
        t[t > 255] = 255
        img[:, :, i] = t
    return img

def coords_transform(x, y, srcode1, srcode2):

    if isinstance(srcode1, int):
        sr1 = CRS.from_epsg(srcode1)
    else:
        sr1 = CRS.from_proj4(srcode1)


    if isinstance(srcode2, int):
        sr2 = CRS.from_epsg(srcode2)
    else:
        sr2 = CRS.from_proj4(srcode2)

    transformer = Transformer.from_crs(sr1, sr2, skip_equivalent=True, always_xy=True)

    res = transformer.transform(x, y)
    return res[0], res[1]


def crop_extent(img_path, dst_path, ulx, uly, lrx, lry, roiepsg=4326):
    if None in [ulx, uly, lrx, lry]:
        return img_path

    if roiepsg == 4326:
        if abs(ulx) > 180:
            ulx = np.sign(ulx) * abs(ulx) % 180
        if abs(lrx) > 180:
            lrx = np.sign(lrx) * abs(lrx) % 180

    print("Crop extent: ", ulx, uly, lrx, lry)
    ds = gdal.Open(img_path, gdal.GA_ReadOnly)
    ncol = ds.RasterXSize
    nrow = ds.RasterYSize
    geoTransform = ds.GetGeoTransform()

    imgsr = osr.SpatialReference()
    imgsr.ImportFromWkt(ds.GetProjectionRef())
    roisr = osr.SpatialReference()
    roisr.ImportFromEPSG(int(roiepsg))

    if roisr.ExportToWkt() != imgsr.ExportToWkt():
        ulx, uly = coords_transform(ulx, uly, roiepsg, imgsr.ExportToProj4())
        lrx, lry = coords_transform(lrx, lry, roiepsg, imgsr.ExportToProj4())
        print("Transformed crop extent: ", ulx, uly, lrx, lry)

    x_ul, y_ul, x_res, y_res = geoTransform[0], geoTransform[3], geoTransform[1], geoTransform[5]
    x_lr = x_ul + x_res * ncol
    y_lr = y_ul + y_res * nrow
    if not (x_ul <= ulx and y_ul >= uly and x_lr >= lrx and y_lr <= lry):
        raise PluginException(1, "crop region exceeds image extent", "裁剪区域超出影像范围")
    ds_crop = gdal.Translate(dst_path, ds, projWin=[ulx, uly, lrx, lry], xRes=x_res, yRes=y_res)
    ds_crop = None
    ds = None
    return dst_path


def get_resolution(source):

    dataset = gdal.Open(source, gdal.GA_ReadOnly)
    geoTransform = dataset.GetGeoTransform()
    sr = dataset.GetSpatialRef()
    if sr is None:
        dataset = None
        return None
    
    if sr.IsGeographic():
    
        nXSize = dataset.RasterXSize  # col
        nYSize = dataset.RasterYSize  # row

        xmin = geoTransform[0]
        ymax = geoTransform[3]
        xmax = xmin + nXSize * geoTransform[1] + nYSize * geoTransform[2]
        ymin = ymax + nXSize * geoTransform[4] + nYSize * geoTransform[5]
        
        source_epsg = sr.GetAuthorityCode('PROJCS')
        if source_epsg is None:
            source_epsg = sr.GetAuthorityCode('GEOGCS')
        if source_epsg is None:
            source_epsg = 4326
        source_epsg = int(source_epsg)
        xmin_3857, ymax_3857 = coords_transform(xmin, ymax, source_epsg, 3857)
        xmax_3857, ymin_3857 = coords_transform(xmax, ymin, source_epsg, 3857)

        x_resolution = (xmax_3857 - xmin_3857) / nXSize
        y_resolution = (ymax_3857 - ymin_3857) / nYSize
    else:
        x_resolution = geoTransform[1]
        y_resolution = -1 * geoTransform[5]
        
    dataset = None

    return (x_resolution + y_resolution) / 2


def raster_to_polygon(src, dst, which_band, mask=False, target_map=None):

    VECTOR_DRIVER = {
        "shp": "ESRI Shapefile",
        "json": "GeoJSON",
        "geojson": "GeoJSON"
    }

    src_ds = gdal.Open(src, gdal.GA_ReadOnly)
    src_band = src_ds.GetRasterBand(which_band)
    sr = osr.SpatialReference()
    sr.ImportFromWkt(src_ds.GetProjectionRef())

    dst_lyrname, dst_ext = os.path.splitext(os.path.basename(dst))
    dst_drvname = VECTOR_DRIVER.get(dst_ext[1:])
    drv = ogr.GetDriverByName(dst_drvname)
    dst_ds = drv.CreateDataSource(dst)
    dst_lyr = dst_ds.CreateLayer(dst_lyrname, sr, ogr.wkbPolygon)

    dst_lyr.CreateField(ogr.FieldDefn("TargetCode", ogr.OFTInteger))
    mask_band = src_band if mask else None
    gdal.Polygonize(src_band, mask_band, dst_lyr, 0, [], callback=None)
    dst_lyr = None
    dst_ds = None
    src_band = None
    src_ds = None

    dst_ds = drv.Open(dst, update=1)
    dst_lyr = dst_ds.GetLayer()
    dst_lyr.CreateField(ogr.FieldDefn("TargetName", ogr.OFTString))
    if target_map is not None:
        dst_lyr.ResetReading()
        for i in range(dst_lyr.GetFeatureCount()):
            fea = dst_lyr.GetFeature(i)
            code = fea.GetField("TargetCode")
            name = target_map.get(int(code))
            if name is None:
                dst_lyr.DeleteFeature(fea.GetFID())
            else:
                fea.SetField("TargetName", name)
                dst_lyr.SetFeature(fea)

    dst_lyr = None
    dst_ds = None

def raster_to_polygon_bfp(src, dst, tmpdir):

    VECTOR_DRIVER = {
        "shp": "ESRI Shapefile",
        "json": "GeoJSON",
        "geojson": "GeoJSON"
    }

    tmp_dst = os.path.join(tmpdir, os.path.basename(dst))

    process_boundary(src, tmp_dst)

    dst_lyrname, dst_ext = os.path.splitext(os.path.basename(dst))
    dst_drvname = VECTOR_DRIVER.get(dst_ext[1:])
    drv = ogr.GetDriverByName(dst_drvname)

    geoms = []

    ds = drv.Open(tmp_dst, update=0)
    lyr = ds.GetLayer()
    crs = lyr.GetSpatialRef()
    
    if crs is None:
        crs_wkt = ''
    else:
        crs_wkt = crs.ExportToWkt()
    for i in range(lyr.GetFeatureCount()):
        fea = lyr.GetFeature(i)
        geom = fea.GetGeometryRef()
        try:
            geom = shapely.wkt.loads(geom.ExportToWkt())
        except Exception as e:
            continue
        if not geom.is_valid:
            geom = geom.convex_hull
        geoms.append(geom)
        fea = None
    lyr = None
    ds = None

    geoms = unary_union(geoms)
    geoms = gpd.GeoSeries(geoms, crs=crs_wkt)
    geoms = geoms.explode()
    
    ds = drv.CreateDataSource(dst)
    lyr = ds.CreateLayer(os.path.splitext(os.path.basename(dst))[0], crs, ogr.wkbPolygon, options=['ENCODING=UTF-8'])
    lyr.CreateField(ogr.FieldDefn("ClsCode", ogr.OFTInteger))
    lyr.CreateField(ogr.FieldDefn("ClsName", ogr.OFTString))
    lyr.CreateField(ogr.FieldDefn("LabelCode", ogr.OFTString))
    lyr.CreateField(ogr.FieldDefn("LabelName", ogr.OFTString))
    lyr_defn = lyr.GetLayerDefn()
    for g in geoms:
        geom = ogr.CreateGeometryFromWkt(shapely.wkt.dumps(g))
        fea = ogr.Feature(lyr_defn)
        fea.SetGeometry(geom)
        lyr.CreateFeature(fea)
        fea = None
    lyr = None
    ds = None

def rescale(srcpath, dstpath, max_length=-1, factor=-1):
    ds = gdal.Open(srcpath, gdal.GA_ReadOnly)
    ncol = ds.RasterXSize
    nrow = ds.RasterYSize
    nband = ds.RasterCount
    print("Input image row, col, band: (%d, %d, %d)" % (nrow, ncol, nband))

    if max_length > 0:
        if nrow * ncol > max_length * max_length:
            factor = float(nrow) / max_length if nrow > ncol else float(ncol) / max_length
            new_nrow, new_ncol = (max_length, int(ncol / factor)) if nrow > ncol else (int(nrow / factor), max_length)
        else:
            new_nrow, new_ncol = nrow, ncol
    elif factor > 0:
        new_nrow, new_ncol = int(nrow * factor), int(ncol * factor)
    else:
        new_nrow, new_ncol = nrow, ncol

    gdal.Translate(dstpath, ds, width=new_ncol, height=new_nrow, resampleAlg=gdal.GRA_NearestNeighbour)
    print("Rescaled shape: (%d, %d, %d)" % (new_nrow, new_ncol, nband))
    ds = None

    return dstpath, nrow, ncol, new_nrow, new_ncol


def assign_spatial_reference_byfile(src_path, dst_path):
    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    sr = osr.SpatialReference()
    sr.ImportFromWkt(src_ds.GetProjectionRef())
    geoTransform = src_ds.GetGeoTransform()
    dst_ds = gdal.Open(dst_path, gdal.GA_Update)
    dst_ds.SetProjection(sr.ExportToWkt())
    dst_ds.SetGeoTransform(geoTransform)
    dst_ds = None
    src_ds = None


def merge_vectors(src_vector_files, dst_path):
    VECTOR_DRIVER = {
        "shp": "ESRI Shapefile",
        "json": "GeoJSON",
        "geojson": "GeoJSON"
    }
    if len(src_vector_files) == 0:
        return None
    src_ext = os.path.splitext(src_vector_files[0])[1]
    src_drv = ogr.GetDriverByName(VECTOR_DRIVER.get(src_ext[1:]))
    src_ds = src_drv.Open(src_vector_files[0])
    src_layer = src_ds.GetLayer()
    sr = src_layer.GetSpatialRef()
    if sr is not None:
        crs = sr.ExportToWkt()
    else:
        crs = ''
        
    src_ds.Destroy()
    
    concat_data = pd.concat([gpd.read_file(src_path) for src_path in src_vector_files]).pipe(gpd.GeoDataFrame)
    geoms = concat_data.geometry.tolist()
    geoms = [g if g.is_valid else g.buffer(0) for g in geoms]

    geoms = unary_union(geoms)
    dst_data = gpd.GeoSeries(geoms, crs=crs)
    dst_data = dst_data.explode()
    dst_data.to_file(dst_path, crs='', encoding='utf-8')

    return dst_path
    

class PluginException(Exception):
    def __init__(self, ret_code, ret_analyse_chn, ret_analyse_eng, err=None):
        if err is None:
            err = ret_analyse_eng
        self.ret_code = ret_code
        self.ret_analyse_eng = ret_analyse_eng
        self.ret_analyse_chn = ret_analyse_chn
        super().__init__(err)


class Fishnet:
    def __init__(self):
        pass

    def Patch2Image(self, coords, patches, image_shape):
        dst = np.zeros(image_shape, dtype=patches[0].dtype)
        for id, (patch, (x, y, w, h)) in enumerate(list(zip(patches, coords))):
            dst[y:(y + h), x:(x + w)] = patch
        return dst

    def Patch2Image_GPU(self, coords, patches, image_shape):
        pass

    def Image2Patch(self, image, step, window_size, out_dir=None, base_name=None):
        '''

        '''
        baseName = None
        if out_dir is None:
            patches = []
        else:
            if base_name is None:
                baseName = ''
            else:
                baseName = base_name
            patches = None
            if not os.path.exists(out_dir) or os.path.isfile(out_dir):
                os.makedirs(out_dir)

        coords = self.__SlidingWindow(image, step, window_size)
        for id, (x, y, w, h) in enumerate(coords):
            patch = image[y:y + h, x:x + w]

            if patches is None:
                fpath = os.path.join(out_dir, baseName + str(id) + '.png')
                self.__WriteRaster(patch, fpath)
            else:
                patches.append(patch)

        if out_dir is None:
            return coords, patches
        else:
            return coords, out_dir

    def __SlidingWindow(self, img, step, window_size):
        '''
        Create sliding window on input image.
        :param img(ndarray): Input image.
        :param step(int): Moving step in pixels.
        :param window_size(tuple of int): width and height of sliding window.
        :return coord(list of (x,y,w,h)): starting column(x), row(y), height, width of sliding windows.
        '''
        # slide a window across the image
        coords = []
        for x in range(0, img.shape[1], step):
            if x + window_size[1] > img.shape[1]:
                x = img.shape[1] - window_size[1]
            for y in range(0, img.shape[0], step):
                if y + window_size[0] > img.shape[0]:
                    y = img.shape[0] - window_size[0]
                coords.append((x, y, window_size[1], window_size[0]))
        return coords

    def __WriteRaster(self, img, out_path):
        io.imsave(out_path, img_as_ubyte(img))

