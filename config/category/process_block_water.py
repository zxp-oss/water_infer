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
from config.utils_category.utils_water import ParamsParser
from config.informations.merge_tif_category import *
from config.informations.read_image import *
from config.informations.satellite_stretch_category import *
from config.informations.other_memory import *
from osgeo import gdal, ogr
import numpy as np
from skimage import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2
import tqdm
import xml.etree.ElementTree as ET
import torch
from torchvision import transforms
import shutil


def do_rs_aux(g, nir, cfg1, cfg2):

    g = g.astype(np.float32)
    n = nir.astype(np.float32)
    sub = g - n
    sum1 = g + n
    aux = cv2.divide(sub, sum1)
    aux[aux < -1] = -1
    aux[aux >  1] =  1
    aux = aux.astype(np.float32)
    pred = np.ones(aux.shape, dtype=np.uint8) * 255
    pred[aux < cfg1] = 0
    pred[aux >= cfg2] = 1

    return pred

def process_one_block_water(image, model, device, params, progress_step):

    # Deep learning inference

    # split image and do inference
    image_optical = image[..., np.array([params.r, params.g, params.b]).astype(np.uint32)-1]
    pad_h = max(params.patch_size - image_optical.shape[0], 0)
    pad_w = max(params.patch_size - image_optical.shape[1], 0)
    image_optical = np.pad(image_optical, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=0)
    
    fn = Fishnet()
    coords, patches = fn.Image2Patch(image_optical, params.patch_stride, (params.patch_size, params.patch_size))
    num_patches = len(patches)
    pred_masks = []
    start_progress = params.progress
    for i in tqdm.tqdm(range(num_patches)):
        
        # prepare inputs
        patch = patches[i].copy()
        patch = patch.astype(np.float32) / 255 * 3.2 - 1.6
        patch = np.expand_dims(patch, 0) # 1, h, w, 3
        patch = patch.transpose([0, 3, 1, 2]) # 1, 3, h, w
        patch = torch.Tensor(patch).to()
        patch = patch.to(device)

        # forward
        with torch.no_grad():
            output = model(patch)
        pred = output.squeeze().cpu().data.numpy()
        pred = (pred >= 0.5).astype(np.uint8)
        pred_masks.append(pred)
        
        params.write_progress(i, (0, num_patches), (start_progress, start_progress + progress_step))

    result = fn.Patch2Image(coords, pred_masks, (image_optical.shape[0], image_optical.shape[1]))
    result = result[0:(result.shape[0]-pad_h), 0:(result.shape[1]-pad_w)]

    # Use rs aux
    if params.use_aux == 1 and image.shape[2]>=4:
        aux_pred = do_rs_aux(image[..., params.g-1], image[..., params.nir-1], params.aux_low, params.aux_high)
        
        # combine rs aux and deep learning results
        result[aux_pred == 0] = 0
        result[aux_pred == 1] = 1

    return result


def process_water(params):

    lg = get_logger()

    # GPU settings
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpu)
    #if params.gpu >= 0:
    
    if torch.cuda.is_available() is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("use device is %s" % device)
    
    # load image
    image_path, res = read_image(params)
    lg.info("Done reading image.")
    params.write_progress(10)

    # load model
    image_name = os.path.basename(image_path).rsplit('.', 1)[0]
    print('========')
    if image_name[0:2].upper() == "GF":
        model_path = params.weights_GF
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        lg.info("Done loading model.")
        params.write_progress(30)
    elif image_name[0:2].upper() == "TH":
        model_path = params.weights_TH
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        lg.info("Done loading model.")
        params.write_progress(30)
    else:
        model_path = params.weights_GF
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        lg.info("Done loading model.")
        params.write_progress(30)


    # split blocks
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    img_width = ds.RasterXSize
    img_height = ds.RasterYSize
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    lg.info("Image nodata found: %s" % str(nodata))
    if nodata is None:
        nodata = 0
    # verify if use aux
    nband = ds.RasterCount
    if params.use_aux is True:
        if nband <= 3:
            params.use_aux = False
            lg.info("USE_AUX requires a minimum of %d bands, got %d bands, set USE_AUX to False" % (4, nband))
        if params.aux_high <= params.aux_low:
            raise PluginException(1, "AUX低阈值必须低于高阈值", "AUX low thr must be lower than high thr")

    block_size = 20000
    if img_height * img_width < block_size * block_size:
        block_coords = [[0, 0, img_width, img_height]]
    else:
        block_coords = get_block_coords(img_height, img_width, block_size, block_size)

    # create output raster
    out_driver = gdal.GetDriverByName('GTiff')
    out_ds = out_driver.Create(params.output_image_path, img_width, img_height, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjectionRef())

    # iterative infer
    num_blocks = len(block_coords)
    progress_step = (80 - 30) // num_blocks
    for i in range(num_blocks):
        lg.info("processing block %d ..." % (i+1))
        # fetch blcok data
        block_coord = block_coords[i]
        block_image = ds.ReadAsArray(xoff=block_coord[0], yoff=block_coord[1], xsize=block_coord[2], ysize=block_coord[3])
        if len(block_image.shape) > 2:
            block_image = np.transpose(block_image, (1, 2, 0))   
        else:
            block_image = np.dstack([block_image, block_image, block_image, block_image])
        # nodata mask
        nodata_mask = np.all(block_image == np.array([nodata,]*nband).reshape(1, 1, nband), axis=2)
        if np.count_nonzero(~nodata_mask) == 0:
            block_result = np.zeros(nodata_mask.shape, dtype=np.uint8)
        else:
            # stretch
            dtype_name = block_image.dtype.name
            if 'int8' in dtype_name:
                if image_name[0:2].upper() == 'TH':
                    block_image = stretch(block_image) 
                elif image_name[0:2].upper() == 'GF':
                    pass
            else:
                block_image = stretch(block_image)
                block_image = block_image.astype(np.uint8)
            # infer on one block
            block_result = process_one_block_water(block_image, model, device, params, progress_step)
            block_result *= 255
            block_result[nodata_mask] = 0
        # write results
        out_ds.WriteRaster(xoff=block_coord[0], yoff=block_coord[1], xsize=block_image.shape[1], ysize=block_image.shape[0],
                           buf_string=block_result.tobytes())
        lg.info("block %d finished" % (i + 1))
    out_ds = None
    lg.info("Done inference on blocks.")

    assign_spatial_reference_byfile(params.input_image_path, params.output_image_path)
    resrange = [params.rangemin, params.rangemax]
    #res = get_resolution(params.input_image_path)
    if res is not None:
        if res <= resrange[0]:
            path2 = r''
            res, rescal_path = rescal_image_back(params, params.output_image_path, path2, params.input_image_path)
            shutil.copy(rescal_path, params.output_image_path)
            assign_spatial_reference_byfile(params.input_image_path, params.output_image_path)
        else:
            pass
    else:
        pass
    # generate browser
    rescale(params.output_image_path, os.path.splitext(params.output_image_path)[0]+"_browser.png", 1024)
    
    # polygonize
    raster_to_polygon(params.output_image_path, params.output_shp_path, 1, True)
    lg.info("Done polygonize.")

    lg.info("Done processing.")