import os
import sys
import logging
import json
from skimage import io, img_as_ubyte
import numpy as np
from pyproj import CRS, Transformer
from osgeo import gdal, ogr, osr
import math
from functools import partial
import uuid
from config.utils import get_rootdir, PluginException


class ParamsParser:
    def __init__(self, filepath):
        self._filepath = filepath
        self.product_level = "水体检测"
        self.product_name = "AI 解译产品"


    def parse(self):
        '''
        :return:
        '''
        with open(self._filepath, "rt", encoding="utf-8") as fp:
            self._obj = json.load(fp)

            # fetch element lists
            configElemList = self._obj.get("Userproperty").get("InputParameter").get("Configuration")
            inputElemList = self._obj.get("Userproperty").get("InputParameter").get("InputFilePath")
            outputElemList = self._obj.get("Userproperty").get("OutputParameter").get("OutputFilePath")

            # parse parameters
            self.progress = 0

            self.is_online = self._get_param_from_list(configElemList, "IsOnline")
            #self.gpu = self._get_param_from_list(configElemList, "GPU", 0)
            self.r = self._get_param_from_list(configElemList, "RedBand", 3)
            self.g = self._get_param_from_list(configElemList, "GreenBand", 2)
            self.b = self._get_param_from_list(configElemList, "BlueBand", 1)
            self.nir = self._get_param_from_list(configElemList, "NirBand", 4)
            self.use_aux = self._get_param_from_list(configElemList, "UseAux")
            self.rangemin = self._get_param_from_list(configElemList, "RangeMin", 3)
            self.rangemax = self._get_param_from_list(configElemList, "RangeMax", 6)
            self.aux_low = self._get_param_from_list(configElemList, "AuxLowThr")
            self.aux_high = self._get_param_from_list(configElemList, "AuxHighThr")

            self.input_image_path = self._get_param_from_list(inputElemList, "InputImgFileName")
            self.output_shp_path = self._get_param_from_list(outputElemList, "OutputVectorFileName")
            self.output_image_path = self._get_param_from_list(outputElemList, "OutputImgFileName")

            self.patch_size = self._get_param_from_list(configElemList, "PatchSize", 1024)
            self.patch_stride = self._get_param_from_list(configElemList, "PatchStride", 1024)

            self.target_name = self._get_param_from_list(configElemList, "TargetName")
            self.weights_GF = os.path.join(get_rootdir(), 'weights', "dlinknet34_water_gpu_GF.pt")
            self.weights_TH = os.path.join(get_rootdir(), 'weights', "dlinknet34_water_gpu_TH.pt")

            # roi
            self.roi_epsg = self._get_param_from_list(configElemList, "ROIEPSG")
            self.xmin = self._get_param_from_list(configElemList, "XMIN")
            self.xmax = self._get_param_from_list(configElemList, "XMAX")
            self.ymin = self._get_param_from_list(configElemList, "YMIN")
            self.ymax = self._get_param_from_list(configElemList, "YMAX")

            # work dir
            self.work_dir = os.path.join(self._obj.get("tmp"), os.path.splitext(os.path.basename(self.input_image_path))[0] + "_" + str(uuid.uuid4())[:8])
            os.makedirs(self.work_dir, exist_ok=True)
            

    def dump(self):
        '''

        :return:
        '''
        with open(self._filepath, "wt", encoding="utf-8") as fp:
            json.dump(self._obj, fp, ensure_ascii=False, indent=4)

    def set_return_info(self, ret_code, ret_analyse_chn, ret_analyse_eng):

        statusList = self._obj['Userproperty']['OutputParameter']['ProgramStatus']
        elem = self._get_element_by_name_from_list(statusList, "ReturnCode")
        elem['value'] = int(ret_code)

        elem = self._get_element_by_name_from_list(statusList, "ReturnAnalyseENG")
        elem['value'] = ret_analyse_eng

        elem = self._get_element_by_name_from_list(statusList, "ReturnAnalyseCHN")
        elem['value'] = ret_analyse_chn


    def write_progress(self, value, source_range=None, target_range=None):
        if isinstance(value, int) or isinstance(value, float):
            value = math.floor(value)
        else:
            raise PluginException(1, "进度值类型错误：%s" % type(value), "Invalid progress data type: %s" % type(value))

        if not source_range is None and not target_range is None:
            value = math.floor((target_range[1] - target_range[0]) * (value - source_range[0]) / (source_range[1] - source_range[0]) + target_range[0])
        if value < 0 or value > 100:
            raise PluginException(1, "无效进度值：%s" % str(value), "Invalid progress value: %s " % str(value))

        statusList = self._obj['Userproperty']['OutputParameter']['ProgramStatus']
        elem = self._get_element_by_name_from_list(statusList, "ProgressInfo")
        if elem is None:
            statusList.append({'name':'ProgressInfo', 'title':'进度信息', 'type':'int', 'value':value})
        else:
            elem['value'] = value

        self.progress = value

        self.dump()

    def _get_param_from_list(self, elem_list, elem_name, default=None, ignore_case=True):

        elem = self._get_element_by_name_from_list(elem_list, elem_name, ignore_case)
        value = self._get_param_from_element(elem)
        if value is None or value == "":
            return default
        return value

    def _get_element_by_name_from_list(self, elem_list, elem_name, ignore_case=True):

        for elem in elem_list:
            name = elem.get("name")
            if not name:
                continue
            name = name.strip()

            if ignore_case:
                flag = elem_name.lower() == name.lower()
            else:
                flag = elem_name == name

            if flag:
                return elem

    def _get_param_from_element(self, elem):
        p_value = None
        try:
            p_raw_value = elem.get("value")
            if not p_raw_value is None:
                if isinstance(p_raw_value, str):
                    p_raw_value = p_raw_value.strip()
                p_type = elem.get("type")
                if p_type == "int":
                    p_value = int(p_raw_value)
                    if p_value == -9999:
                        p_value = None
                elif p_type == "float":
                    p_value = float(p_raw_value)
                    if p_value == -9999:
                        p_value = None
                elif p_type == "string":
                    p_value = str(p_raw_value)
                elif p_type == "url":
                    p_value = str(p_raw_value)
                elif p_type == "select":
                    p_value = p_raw_value
        except Exception as e:
            print(str(e))
        finally:
            return p_value