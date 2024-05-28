import io
import os
import warnings
import sys

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

import cv2
import skimage
from skimage import morphology
import numpy as np
import math
import shapely
from shapely.geometry import Polygon
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

VECTOR_DRIVER = {
    "shp": "ESRI Shapefile",
    "json": "GeoJSON",
    "geojson": "GeoJSON"
}

try:
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
except ImportError:
    import gdal
    import ogr
    import osr


def spin(approx, contour, contour_area, startx, starty, endx, endy, vec_se, seg_type, h):
    l1 = cv2.arcLength(np.round(np.array(approx)).astype(np.int32), True) / len(approx)
    skip = 0
    show = False
    appro = approx.copy()
    for i in range(len(approx) - 1, -1, -1):
        if len(approx) < 4: break
        for f in range(i + 2, i - 3, -1):
            if all(np.round(approx[f % len(approx)]) == np.round(approx[(f - 1) % len(approx)])):
                if 0 <= f % len(approx) <= i:
                    approx.pop(f % len(approx))
                    skip = True
                    break
                else:
                    approx.pop(f % len(approx))
        if skip or len(approx) < 4:
            skip = 0
            continue
        x0, y0 = approx[(i + 1) % len(approx)];
        x1, y1 = approx[i];
        x2, y2 = approx[i - 1];
        x3, y3 = approx[i - 2]
        vec_01, vec_12, vec_23 = (x1 - x0, y1 - y0), (x2 - x1, y2 - y1), (x3 - x2, y3 - y2)
        dis_01, dis_12, dis_23 = math.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2), math.sqrt(
            (y1 - y2) ** 2 + (x1 - x2) ** 2), math.sqrt((y3 - y2) ** 2 + (x3 - x2) ** 2)
        angle1, angle2, angle_01, angle_12, angle_23, angle_12x, angle_sex = cal_angle(vec_01, vec_12,
                                                                                       False), cal_angle(vec_12, vec_23,
                                                                                                         False), cal_angle(
            vec_01, vec_se), cal_angle(vec_12, vec_se), cal_angle(vec_23, vec_se), cal_angle(vec_12, (1, 0)), cal_angle(
            vec_se, (1, 0))
        min01, min12, min23, min12x, minsex = min(angle_01, 90 - angle_01), min(angle_12, 90 - angle_12), min(angle_23,
                                                                                                              90 - angle_23), min(
            angle_12x, 90 - angle_12x), min(angle_sex, 90 - angle_sex)
        app0, app1, app2, app3 = approx.copy(), approx.copy(), approx.copy(), approx.copy();
        app0.pop((i + 1) % len(approx));
        app1.pop(i);
        app2.pop(i - 1);
        app3.pop(i - 2)
        h1, _ = horv(approx, i, vec_12, dis_12, l1, False)
        x_y1, x_y2 = max(1.5 * x1 - 0.5 * x0, x1 + 13 * (x1 - x0) / dis_01), max(1.5 * x2 - 0.5 * x3,
                                                                                 x2 + 13 * (x2 - x3) / dis_23)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        conti = 0;
        con = 0
        A, B, C = two_points_line(startx, starty, endx, endy, cx, cy) if angle_12 <= 45 else two_points_vline(startx,
                                                                                                              starty,
                                                                                                              endx,
                                                                                                              endy, cx,
                                                                                                              cy)
        x_d1, y_d1 = Drop_point(A, B, C, x1, y1);
        x_d2, y_d2 = Drop_point(A, B, C, x2, y2)
        angle_1, angle_2 = cal_angle(vec_01, (x_d2 - x_d1, y_d2 - y_d1), False), cal_angle((x_d2 - x_d1, y_d2 - y_d1),
                                                                                           vec_23, False)
        x_1, y_1 = findIntersection(x_d1, y_d1, x_d2, y_d2, x0, y0, x1, y1);
        x_2, y_2 = findIntersection(x_d1, y_d1, x_d2, y_d2, x2, y2, x3, y3)
        angle_0_1, angle__23 = cal_angle((x_d1 - x0, y_d1 - y0), vec_se), cal_angle((x3 - x_d2, y3 - y_d2), vec_se)
        min0_1, min_23 = min(angle_0_1, 90 - angle_0_1), min(angle__23, 90 - angle__23)
        condition = (min12x >= 5 or dis_12 <= 15 or minsex <= min12x + 3) \
                    and (0 < angle_12 < 12 \
                         or angle_12 < 20 and not h1 \
                         or 82 < angle_12 < 90 \
                         or angle_12 > 75 and not h1 \
                         or dis_12 <= 15 and (angle_12 < 25 or angle_12 > 65) \
                         or (min(angle_1, abs(angle_1 - 90)) <= min(angle1, abs(angle1 - 90)) or min0_1 <= min01 and (
                        angle_12 < 20 or angle_12 > 70 or dis_12 <= 15)) and (
                                 min(angle_2, abs(angle_2 - 90)) <= min(angle2,
                                                                        abs(angle2 - 90)) or min_23 <= min23 and (
                                         angle_12 < 20 or angle_12 > 70 or dis_12 <= 15)) \
                         or rayCasting([x2, y2], app2) and rayCasting([x3, y3], app3) and (
                                 min(angle_1, abs(angle_1 - 90)) <= min(angle1,
                                                                        abs(angle1 - 90)) or min0_1 <= min01) \
                         or rayCasting([x1, y1], app1) and rayCasting([x2, y2], app2) \
                         or rayCasting([x0, y0], app0) and rayCasting([x1, y1], app1) and (
                                 min(angle_2, abs(angle_2 - 90)) <= min(angle2,
                                                                        abs(angle2 - 90)) or min_23 <= min23))
        if min12 >= 15:
            h1, v1 = horv(approx, i, vec_12, dis_12, l1, True)
            condition = condition and not h1 and not v1
        if condition:
            if (x_1 == None or (x_1 - x0) * (x0 - x1) > 0 or (x_1 - x_y1) * (x_y1 - x1) >= 0) and not (
                    x_2 == None or (x_2 - x3) * (x3 - x2) > 0 or (x_2 - x_y2) * (x_y2 - x2) >= 0) \
                    and min0_1 <= min01 and area_judg(
                polygon_area([[x1, y1], [x_d1, y_d1], [cx, cy]]) + polygon_area([[x2, y2], [x_2, y_2], [cx, cy]]),
                contour_area, 1, seg_type):
                approx[i] = [x_d1, y_d1]
                approx[i - 1] = [x_2, y_2]
            elif not (x_1 == None or (x_1 - x0) * (x0 - x1) > 0 or (x_1 - x_y1) * (x_y1 - x1) >= 0) and (
                    x_2 == None or (x_2 - x3) * (x3 - x2) > 0 or (x_2 - x_y2) * (x_y2 - x2) >= 0) \
                    and min_23 <= min23 and area_judg(
                polygon_area([[x1, y1], [x_1, y_1], [cx, cy]]) + polygon_area([[x2, y2], [x_d2, y_d2], [cx, cy]]),
                contour_area, 1, seg_type):
                approx[i] = [x_1, y_1]
                approx[i - 1] = [x_d2, y_d2]
            elif not (x_1 == None or x_2 == None or (x_1 - x0) * (x0 - x1) > 0 or (x_1 - x_2) * (x_2 - cx) >= 0 or (
                    x_1 - x_y1) * (x_y1 - x1) >= 0 or (x_2 - x3) * (x3 - x2) > 0 or (x_2 - x_1) * (x_1 - cx) >= 0 or (
                              x_2 - x_y2) * (x_y2 - x2) >= 0) \
                    and area_judg(
                polygon_area([[x1, y1], [x_1, y_1], [cx, cy]]) + polygon_area([[x2, y2], [x_2, y_2], [cx, cy]]),
                contour_area, 1, seg_type):
                approx[i] = [x_1, y_1]
                approx[i - 1] = [x_2, y_2]
            elif x_1 == None or x_2 == None or (x_1 - x0) * (x0 - x1) > 0 or (x_1 - x_2) * (x_2 - cx) >= 0 or (
                    x_1 - x_y1) * (x_y1 - x1) >= 0 or (x_2 - x3) * (x3 - x2) > 0 or (x_2 - x_1) * (x_1 - cx) >= 0 or (
                    x_2 - x_y2) * (x_y2 - x2) >= 0:
                if min0_1 <= min01 and min_23 <= min23 and area_judg(
                        polygon_area([[x1, y1], [x_d1, y_d1], [cx, cy]]) + polygon_area(
                            [[x2, y2], [x_d2, y_d2], [cx, cy]]), contour_area, 1, seg_type):
                    approx[i] = [x_d1, y_d1]
                    approx[i - 1] = [x_d2, y_d2]
        # if not h:
        # show_result(contour,approx,approx[i][0],approx[i][1],True)
    return approx, show


def area_judg(area, contour_area, index, seg_type):
    b = 10
    a = 2
    c = a / 2
    if index == -4:
        return area <= 55 / c and area / contour_area <= 0.05 / a or area / contour_area <= 0.02 / a if seg_type == 'bfp' else area <= 55 and area / contour_area <= 0.05 / b or area / contour_area <= 0.02 / b / 2
    elif index == -3:
        return area <= 55 / c and area / contour_area <= 0.05 / a if seg_type == 'bfp' else area <= 55 and area / contour_area <= 0.05 / b
    elif index == -2:
        return area <= 55 / c and area / contour_area <= 0.1 / a if seg_type == 'bfp' else area <= 55 and area / contour_area <= 0.1 / b
    elif index == -1:
        return area <= 55 / c and area / contour_area <= 0.015 / a or area / contour_area <= 0.006 / a if seg_type == 'bfp' else area <= 55 and area / contour_area <= 0.015 / b or area / contour_area <= 0.006 / b / 2
    elif index == 0:
        return area <= 55 / c and area / contour_area <= 0.1 / a or area / contour_area <= 0.04 / a if seg_type == 'bfp' else area <= 55 and area / contour_area <= 0.1 / b or area / contour_area <= 0.04 / b / 2
    elif index == 1:
        return area <= 85 / c and area / contour_area <= 0.15 / a or area / contour_area <= 0.06 / a if seg_type == 'bfp' else area <= 85 and area / contour_area <= 0.15 / b or area / contour_area <= 0.06 / b / 2
    elif index == 2:
        return area <= 110 / c and area / contour_area <= 0.2 / a or area / contour_area <= 0.08 / a if seg_type == 'bfp' else area <= 110 and area / contour_area <= 0.2 / b or area / contour_area <= 0.08 / b / 2
    elif index == 3:
        return area <= 110 / c and area / contour_area <= 0.2 / a or area / contour_area <= 0.1 / a if seg_type == 'bfp' else area <= 110 and area / contour_area <= 0.2 / b or area / contour_area <= 0.1 / b / 2
    elif index == 4:
        return area <= 150 / c and area / contour_area <= 0.25 / a or area / contour_area <= 0.1 / a if seg_type == 'bfp' else area <= 150 and area / contour_area <= 0.25 / b or area / contour_area <= 0.1 / b / 2
    elif index == 5:
        return area <= 200 / c and area / contour_area <= 0.3 / a or area / contour_area <= 0.12 / a if seg_type == 'bfp' else area <= 200 and area / contour_area <= 0.3 / b or area / contour_area <= 0.12 / b / 2
    elif index == 6:
        return area / contour_area <= 0.5 if seg_type == 'bfp' else area / contour_area <= 0.5 / b


def fill(approx, contour_area, startx, starty, endx, endy, vec_se, seg_type):
    if len(approx) < 5:
        return approx
    app = approx.copy()
    app = np.round(np.array(app)).astype(np.int32).reshape(-1, 1, 2)
    for i in range(len(approx) - 1, -1, -1):
        if approx[(i + 1) % len(approx)] == approx[i] or approx[i] == approx[i - 1]:
            approx.pop(i)
            continue
        x_, y_ = approx[(i + 2) % len(approx)]
        x0, y0 = approx[(i + 1) % len(approx)]
        x1, y1 = approx[i]
        x2, y2 = approx[i - 1]
        x3, y3 = approx[i - 2]
        dis_01, dis_12, dis_02 = math.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2), math.sqrt(
            (y1 - y2) ** 2 + (x1 - x2) ** 2), math.sqrt((y2 - y0) ** 2 + (x2 - x0) ** 2)  # 01线长
        try:
            angle0, angle, angle2 = cal_ang([x_, y_], [x0, y0], [x1, y1]), cal_ang([x0, y0], [x1, y1],
                                                                                   [x2, y2]), cal_ang([x1, y1],
                                                                                                      [x2, y2],
                                                                                                      [x3, y3])  # 三点夹角
        except:
            continue
        vec__0, vec_01, vec_12, vec_23, vec_02 = (x0 - x_, y0 - y_), (x1 - x0, y1 - y0), (x2 - x1, y2 - y1), (
            x3 - x2, y3 - y2), (x2 - x0, y2 - y0)  # 01,12向量
        angle__0, angle_01, angle_12, angle_23, angle_02 = cal_angle(vec__0, vec_se), cal_angle(vec_01,
                                                                                                vec_se), cal_angle(
            vec_12, vec_se), cal_angle(vec_23, vec_se), cal_angle(vec_02, vec_se)  # 01,02与主方向夹角
        min20 = min(angle_01, 90 - angle_01, angle_12, 90 - angle_12)
        appr = approx.copy();
        appr.pop(i)
        area_1 = polygon_area([[x0, y0], [x1, y1], [x2, y2]])
        if dis_01 <= 3 and abs(90 - angle0) > 20 or dis_12 <= 3 and abs(90 - angle2) > 20 and abs(90 - angle) > 20:
            if dis_01 <= 3:
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                approx[(i + 1) % len(approx)] = [cx, cy]
            if dis_12 <= 3:
                cx, cy = (x2 + x1) / 2, (y2 + y1) / 2
                approx[i - 1] = [cx, cy]
            approx.pop(i)
            continue
        elif angle >= 165 or angle >= 160 and min20 <= min(min(angle_01, 90 - angle_01), min(angle_12, 90 - angle_12)):
            cx_01, cy_01 = (x0 + x1) / 2, (y0 + y1) / 2
            cx_12, cy_12 = (x2 + x1) / 2, (y2 + y1) / 2
            Ac, Bc, Cc = two_points_line(cx_01, cy_01, cx_12, cy_12, cx_01, cy_01)
            x_0, y_0 = Drop_point(Ac, Bc, Cc, x0, y0)
            x_2, y_2 = Drop_point(Ac, Bc, Cc, x2, y2)
            approx[(i + 1) % len(approx)] = [x_0, y_0]
            approx[i - 1] = [x_2, y_2]
            approx.pop(i)
        elif angle <= 30:
            c_x_12, c_y_12, c_x_01, c_y_01, area_c_12, area_c_01, intersect_12, intersect_01 = intersect(x_, y_, x0, y0,
                                                                                                         x1, y1, x2, y2,
                                                                                                         x3, y3, True)
            if intersect_12 and math.sqrt((y2 - c_y_12) ** 2 + (x2 - c_x_12) ** 2) >= 5 and area_judg(area_c_12,
                                                                                                      contour_area, 2,
                                                                                                      seg_type):
                approx[(i + 1) % len(approx)] = [c_x_12, c_y_12]
                approx.pop(i)
            elif intersect_01 and math.sqrt((y0 - c_y_01) ** 2 + (x0 - c_x_01) ** 2) >= 5 and area_judg(area_c_01,
                                                                                                        contour_area, 2,
                                                                                                        seg_type):
                approx[i - 1] = [c_x_01, c_y_01]
                approx.pop(i)
            elif area_judg(area_1, contour_area, 2, seg_type):
                approx.pop(i)
        elif (angle <= 50 or angle <= 70 and min20 > 15) and (area_judg(area_1, contour_area, -1, seg_type) or (
                dis_01 < 4 or dis_12 < 4 or dis_01 + dis_12 < 14) and area_judg(area_1, contour_area, 1, seg_type)):
            c_x_12, c_y_12, c_x_01, c_y_01, area_c_12, area_c_01, intersect_12, intersect_01 = intersect(x_, y_, x0, y0,
                                                                                                         x1, y1, x2, y2,
                                                                                                         x3, y3, True)
            if intersect_12 and math.sqrt((y2 - c_y_12) ** 2 + (x2 - c_x_12) ** 2) > 4:
                approx[(i + 1) % len(approx)] = [c_x_12, c_y_12]
            elif intersect_01 and math.sqrt((y0 - c_y_01) ** 2 + (x0 - c_x_01) ** 2) > 4:
                approx[i - 1] = [c_x_01, c_y_01]
            approx.pop(i)
    return approx


def del_no_feature_process(no_feature, angle_13, angle_12, angle_34, minx, miny, contour_area, vec_se, startx, starty,
                           endx, endy, approx, l1, seg_type):
    process = False
    for i in range(len(no_feature[1:-2])):
        xi, yi = no_feature[i + 1];
        x_i, y_i = no_feature[i + 2]
        angle_is = cal_angle((x_i - xi, y_i - yi), vec_se)
        if 10 <= angle_is <= 80:
            process = True
    if not process or no_feature[0] == no_feature[-1]:
        return no_feature, False, no_feature[0][0], no_feature[0][1]
    else:
        total = len(no_feature)
        x1, y1 = no_feature[0];
        x2, y2 = no_feature[1];
        x3, y3 = no_feature[-2];
        x4, y4 = no_feature[-1]

        try:
            angle2, angle3 = cal_ang((x1, y1), (x2, y2), no_feature[2]), cal_ang((x4, y4), (x3, y3), no_feature[-3])
        except:
            for f in range(len(no_feature) - 1, -1, -1):
                if all(np.round(no_feature[f]) == np.round(no_feature[f - 1])):
                    no_feature.pop(f)
            return no_feature, False, no_feature[0][0], no_feature[0][1]

        angle_23, angle_12x, angle_34x = cal_angle((x3 - x2, y3 - y2), vec_se), cal_angle((x2 - x1, y2 - y1),
                                                                                          (1, 0)), cal_angle(
            (x4 - x3, y4 - y3), (1, 0))
        min12, min23, min34, min12x, min34x = min(angle_12 - 0, 90 - angle_12), min(angle_23 - 0, 90 - angle_23), min(
            angle_34 - 0, 90 - angle_34), min(angle_12x - 0, 90 - angle_12x), min(angle_34x - 0, 90 - angle_34x)
        app2, app3 = approx.copy(), approx.copy();
        app2.remove([x2, y2]);
        app3.remove([x3, y3])
        angle_1m, angle_4m = cal_angle((no_feature[2][0] - no_feature[0][0], no_feature[2][1] - no_feature[0][1]),
                                       vec_se), cal_angle(
            (no_feature[-1][0] - no_feature[-3][0], no_feature[-1][1] - no_feature[-3][1]), vec_se)
        min1m, min4m = min(angle_1m - 0, 90 - angle_1m), min(angle_4m - 0, 90 - angle_4m)
        if total == 4:
            min2m = min3m = min23
        elif total == 5:
            midx, midy = no_feature[2]
            angle_2m, angle_3m = cal_angle((midx - x2, midy - y2), vec_se), cal_angle((x3 - midx, y3 - midy), vec_se)
            min2m, min3m = min(angle_2m - 0, 90 - angle_2m), min(angle_3m - 0, 90 - angle_3m)
            appf2 = approx.copy();
            appf2.remove([midx, midy])
        elif total == 6:
            xf2, yf2 = no_feature[2];
            xf3, yf3 = no_feature[3]
            angle_2m, angle_3m, angle_mm, angle_2mf, angle_3mf = cal_angle((xf2 - x2, yf2 - y2), vec_se), cal_angle(
                (x3 - xf3, y3 - yf3), vec_se), cal_angle((xf3 - xf2, yf3 - yf2), vec_se), cal_angle(
                (xf3 - x2, yf3 - y2), vec_se), cal_angle((x3 - xf2, y3 - yf2), vec_se)
            min2m, min3m, minmm, min2mf, min3mf = min(angle_2m - 0, 90 - angle_2m), min(angle_3m - 0,
                                                                                        90 - angle_3m), min(
                angle_mm - 0, 90 - angle_mm), min(angle_2mf - 0, 90 - angle_2mf), min(angle_3mf - 0, 90 - angle_3mf)
            appf2, appf3 = approx.copy(), approx.copy();
            appf2.remove([xf2, yf2]);
            appf3.remove([xf3, yf3])

        if rayCasting([x2, y2], app2) and min1m <= min(min12, min2m, 7) and (
                angle_13 < 170 or total > 4) and min12x > 4 and area_judg(polygon_area(no_feature[:3]), contour_area, 1,
                                                                          seg_type) and abs(90 - angle2) > 15:
            no_feature.pop(1)
        elif rayCasting([x3, y3], app3) and min4m <= min(min34, min3m, 7) and (
                angle_13 < 170 or total > 4) and min34x > 4 and area_judg(polygon_area(no_feature[-3:]), contour_area,
                                                                          1, seg_type) and abs(90 - angle3) > 15:
            no_feature.pop(-2)

        if len(no_feature) == 5:
            if total == 6:
                x2, y2 = no_feature[1];
                x3, y3 = no_feature[-2];
                midx, midy = no_feature[2]
                angle_2m, angle_3m, angle_23 = cal_angle((midx - x2, midy - y2), vec_se), cal_angle(
                    (x3 - midx, y3 - midy), vec_se), cal_angle((x3 - x2, y3 - y2), vec_se)
                min2m, min3m, min23 = min(angle_2m - 0, 90 - angle_2m), min(angle_3m - 0, 90 - angle_3m), min(
                    angle_23 - 0, 90 - angle_23)
                appf2 = approx.copy();
                appf2.remove([midx, midy])
            if rayCasting(no_feature[2], appf2) and min23 <= min(min2m, min3m, 7) and area_judg(
                    polygon_area(no_feature[1:-1]), contour_area, 1, seg_type):
                no_feature.pop(2)

        elif len(no_feature) == 6:
            if rayCasting([xf2, yf2], appf2) and min2mf <= min(min2m, minmm, 7) and area_judg(
                    polygon_area(no_feature[1:-2]), contour_area, 2, seg_type):
                no_feature.pop(2)
            elif rayCasting([xf3, yf3], appf3) and min3mf <= min(min3m, minmm, 7) and area_judg(
                    polygon_area(no_feature[2:-1]), contour_area, 2, seg_type):
                no_feature.pop(-3)

        angle_13 = cal_angle((no_feature[1][0] - no_feature[0][0], no_feature[1][1] - no_feature[0][1]),
                             (no_feature[-1][0] - no_feature[-2][0], no_feature[-1][1] - no_feature[-2][1]), half=False)
        if len(no_feature) > 3 and (angle_13 <= 10 or angle_13 >= 170 or 80 <= angle_13 <= 100):
            no_feature, show, x, y = no_feature_process(no_feature, angle_13, minx, miny, contour_area, vec_se, startx,
                                                        starty, endx, endy, approx, l1, seg_type)
            return no_feature, show, x, y
        else:
            return no_feature, False, no_feature[0][0], no_feature[0][1]


def no_feature_process(no_feature, angle_13, minx, miny, contour_area, vec_se, startx, starty, endx, endy, approx, l1,
                       seg_type):
    show = False
    x1, y1 = no_feature[0];
    x2, y2 = no_feature[1];
    x3, y3 = no_feature[-2];
    x4, y4 = no_feature[-1]
    dis_12, dis_23, dis_34 = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2), math.sqrt(
        (y3 - y2) ** 2 + (x3 - x2) ** 2), math.sqrt((y4 - y3) ** 2 + (x4 - x3) ** 2)
    angle2, angle3, angle_12, angle_23, angle_34, angle_12x, angle_34x = cal_ang((x1, y1), (x2, y2),
                                                                                 no_feature[2]), cal_ang((x4, y4),
                                                                                                         (x3, y3),
                                                                                                         no_feature[
                                                                                                             -3]), cal_angle(
        (x2 - x1, y2 - y1), vec_se), cal_angle((x3 - x2, y3 - y2), vec_se), cal_angle((x4 - x3, y4 - y3),
                                                                                      vec_se), cal_angle(
        (x2 - x1, y2 - y1), (1, 0)), cal_angle((x4 - x3, y4 - y3), (1, 0))
    min12, min23, min34, min12x, min34x = min(angle_12 - 0, 90 - angle_12), min(angle_23 - 0, 90 - angle_23), min(
        angle_34 - 0, 90 - angle_34), min(angle_12x - 0, 90 - angle_12x), min(angle_34x - 0, 90 - angle_34x)
    A1, B1, C1 = two_points_line(x1, y1, x2, y2, x1, y1);
    A3, B3, C3 = two_points_line(x3, y3, x4, y4, x3, y3)
    d4, d3, d2 = abs(A1 * x4 + B1 * y4 + C1) / math.sqrt(A1 * A1 + B1 * B1), abs(A1 * x3 + B1 * y3 + C1) / math.sqrt(
        A1 * A1 + B1 * B1), abs(A3 * x2 + B3 * y2 + C3) / math.sqrt(A3 * A3 + B3 * B3)
    app2, app3 = approx.copy(), approx.copy();
    app2.remove([x2, y2]);
    app3.remove([x3, y3])
    angle_1m, angle_4m = cal_angle((no_feature[2][0] - no_feature[0][0], no_feature[2][1] - no_feature[0][1]),
                                   vec_se), cal_angle(
        (no_feature[-1][0] - no_feature[-3][0], no_feature[-1][1] - no_feature[-3][1]), vec_se);
    min1m, min4m = min(angle_1m - 0, 90 - angle_1m), min(angle_4m - 0, 90 - angle_4m)
    total = len(no_feature)
    if total == 5:
        anglem = cal_ang((x2, y2), no_feature[2], (x3, y3))
        midx, midy = no_feature[2]
        c_x_2m, c_y_2m, c_x_3m, c_y_3m, area_c_2m, area_c_3m, intersect_2m, intersect_3m = intersect(x4, y4, x3, y3,
                                                                                                     midx, midy, x2, y2,
                                                                                                     x1, y1, False)
        appf2 = approx.copy();
        appf2.remove([midx, midy])
        angle_2m, angle_3m = cal_angle((midx - x2, midy - y2), vec_se), cal_angle((x3 - midx, y3 - midy), vec_se);
        min2m, min3m = min(angle_2m - 0, 90 - angle_2m), min(angle_3m - 0, 90 - angle_3m)
    elif total == 6:
        xf2, yf2 = no_feature[2];
        xf3, yf3 = no_feature[3]
        vec_2m, vec_3m = (xf2 - x2, yf2 - y2), (x3 - xf3, y3 - yf3)
        dis_2m, dis_mm, dis_3m = math.sqrt((xf2 - x2) ** 2 + (yf2 - y2) ** 2), math.sqrt(
            (xf3 - xf2) ** 2 + (yf3 - yf2) ** 2), math.sqrt((x3 - xf3) ** 2 + (y3 - yf3) ** 2)
        angle_2m, angle_3m, angle_mm, angle_2m3 = cal_angle(vec_2m, vec_se), cal_angle(vec_3m, vec_se), cal_angle(
            (xf3 - xf2, yf3 - yf2), vec_se), cal_angle(vec_2m, vec_3m)
        min2m, min3m, minmm = min(angle_2m - 0, 90 - angle_2m), min(angle_3m - 0, 90 - angle_3m), min(angle_mm - 0,
                                                                                                      90 - angle_mm)
        appf2, appf3 = approx.copy(), approx.copy();
        appf2.remove([xf2, yf2]);
        appf3.remove([xf3, yf3])
        df2_1, df2_3, df3_1, df3_3 = abs(A1 * xf2 + B1 * yf2 + C1) / math.sqrt(A1 * A1 + B1 * B1), abs(
            A3 * xf2 + B3 * yf2 + C3) / math.sqrt(A3 * A3 + B3 * B3), abs(A1 * xf3 + B1 * yf3 + C1) / math.sqrt(
            A1 * A1 + B1 * B1), abs(A3 * xf3 + B3 * yf3 + C3) / math.sqrt(A3 * A3 + B3 * B3)

    if angle_13 <= 10:
        # show=True
        conti = False
        if total == 4:
            conti = 1 if d2 + d3 >= 10 else 2
        elif total == 5:
            if rayCasting(no_feature[2], appf2):
                if intersect_2m and area_judg(area_c_2m, contour_area, 2, seg_type):
                    no_feature[-2] = [c_x_2m, c_y_2m]
                    no_feature.pop(2)
                    d3 = abs(A1 * c_x_2m + B1 * c_y_2m + C1) / math.sqrt(A1 * A1 + B1 * B1)
                    conti = 1 if d2 + d3 >= 8 else 2
                elif intersect_3m and area_judg(area_c_3m, contour_area, 2, seg_type):
                    no_feature[1] = [c_x_3m, c_y_3m]
                    no_feature.pop(2)
                    d2 = abs(A3 * c_x_3m + B3 * c_y_3m + C3) / math.sqrt(A3 * A3 + B3 * B3)
                    conti = 1 if d2 + d3 > 8 else 2
                elif not intersect_2m and not intersect_3m:
                    dm1, dm3 = abs(A1 * midx + B1 * midy + C1) / math.sqrt(A1 * A1 + B1 * B1), abs(
                        A3 * midx + B3 * midy + C3) / math.sqrt(A3 * A3 + B3 * B3)
                    if d2 + d3 > 8:
                        conti = 3
                    elif dm1 < (d2 + d3) / 2 and dm3 < (d2 + d3) / 2:
                        no_feature.pop(2)
                        conti = 2
                    else:
                        conti = 4
            else:
                if intersect_2m:
                    if math.sqrt((midy - c_y_2m) ** 2 + (midx - c_x_2m) ** 2) > 4:
                        conti = 4
                    else:
                        no_feature[-2] = [c_x_2m, c_y_2m]
                        no_feature.pop(2)
                        d3 = abs(A1 * c_x_2m + B1 * c_y_2m + C1) / math.sqrt(A1 * A1 + B1 * B1)
                        conti = 1 if d2 + d3 > 8 else 2
                elif intersect_3m:
                    if math.sqrt((midy - c_y_3m) ** 2 + (midx - c_x_3m) ** 2) > 4:
                        conti = 4
                    else:
                        no_feature[1] = [c_x_3m, c_y_3m]
                        no_feature.pop(2)
                        d2 = abs(A3 * c_x_3m + B3 * c_y_3m + C3) / math.sqrt(A3 * A3 + B3 * B3)
                        conti = 1 if d2 + d3 > 8 else 2
                else:
                    dm1, dm3 = abs(A1 * midx + B1 * midy + C1) / math.sqrt(A1 * A1 + B1 * B1), abs(
                        A3 * midx + B3 * midy + C3) / math.sqrt(A3 * A3 + B3 * B3)
                    if d2 + d3 > 8:
                        conti = 3
                    elif dm1 < (d2 + d3) / 2 and dm3 < (d2 + d3) / 2:
                        no_feature.pop(2)
                        conti = 2
                    else:
                        conti = 4
        elif rayCasting([xf2, yf2], appf2) == rayCasting([xf3, yf3], appf3) and rayCasting([x2, y2],
                                                                                           app2) == rayCasting([x3, y3],
                                                                                                               app3):
            if angle_2m3 <= 10 and dis_2m + dis_3m >= 1.6 * dis_mm and minmm > 10:  # abs(90-cal_ang((x2,y2),(xf2,yf2),(xf3,yf3)))>20
                A2m, B2m, C2m = two_points_line(x2, y2, xf2, yf2, x2, y2)
                A3m, B3m, C3m = two_points_line(x3, y3, xf3, yf3, x3, y3)
                cx, cy = (xf2 + xf3) / 2, (yf2 + yf3) / 2
                xf_2, yf_2 = Drop_point(A2m, B2m, C2m, cx, cy)
                xf_3, yf_3 = Drop_point(A3m, B3m, C3m, cx, cy)
                area_c = polygon_area([[xf_2, yf_2], [cx, cy], [xf2, yf2]]) + polygon_area(
                    [[xf_3, yf_3], [cx, cy], [xf3, yf3]])
                if area_judg(area_c, contour_area, 1, seg_type):
                    no_feature[2] = [xf_2, yf_2]
                    no_feature[-3] = [xf_3, yf_3]
            elif angle_2m3 > 10 and dis_12 + dis_34 >= 0.5 * (dis_2m + dis_mm + dis_3m):
                df, far = 0, None
                for i in no_feature[2:-2]:
                    di = max(abs(A1 * i[0] + B1 * i[1] + C1) / math.sqrt(A1 * A1 + B1 * B1),
                             abs(A3 * i[0] + B3 * i[1] + C3) / math.sqrt(A3 * A3 + B3 * B3))
                    if di > df:
                        far = i
                        df = di
                Ac, Bc, Cc = two_points_line(x1, y1, x2, y2, far[0], far[1]) if min12 <= min34 else two_points_line(x3,
                                                                                                                    y3,
                                                                                                                    x4,
                                                                                                                    y4,
                                                                                                                    far[
                                                                                                                        0],
                                                                                                                    far[
                                                                                                                        1])
                cx2, cy2, cx3, cy3 = (x2 + xf2) / 2, (y2 + yf2) / 2, (x3 + xf3) / 2, (y3 + yf3) / 2
                x_2, y_2 = Drop_point(A1, B1, C1, cx2, cy2)
                xf_2, yf_2 = Drop_point(Ac, Bc, Cc, cx2, cy2)
                xf_3, yf_3 = Drop_point(Ac, Bc, Cc, cx3, cy3)
                x_3, y_3 = Drop_point(A3, B3, C3, cx3, cy3)
                area_c = union(no_feature[1:-1], [[x_2, y_2], [xf_2, yf_2], [xf_3, yf_3], [x_3, y_3]])
                if area_judg(area_c, contour_area, 2, seg_type):
                    no_feature[1:-1] = [x_2, y_2], [xf_2, yf_2], [xf_3, yf_3], [x_3, y_3]
                    dis_2m, dis_3m = math.sqrt((xf_2 - x_2) ** 2 + (yf_2 - y_2) ** 2), math.sqrt(
                        (xf_3 - x_3) ** 2 + (yf_3 - y_3) ** 2)
                    if 3 < dis_2m <= 7 and 3 < dis_3m <= 7:
                        pass
                    else:
                        if dis_2m <= 3.5:
                            cx, cy = (x_2 + xf_2) / 2, (y_2 + yf_2) / 2
                            x__2, y__2 = cx + (x_2 - x1), cy + (y_2 - y1)
                            x__4, y__4 = cx + (xf_3 - xf_2), cy + (yf_3 - yf_2)
                            c_x, c_y = (x__2 + x__4) / 2, (y__2 + y__4) / 2
                            xf__3, yf__3 = findIntersection(cx, cy, c_x, c_y, x_3, y_3, xf_3, yf_3)
                            if xf__3:
                                xf_3, yf_3 = xf__3, yf__3;
                                dis_3m = math.sqrt((xf_3 - x_3) ** 2 + (yf_3 - y_3) ** 2)
                                no_feature[1] = [xf_3, yf_3]
                                no_feature.pop(2)
                                no_feature.pop(-3)
                        if dis_3m <= 3.5:
                            if len(no_feature) == 4:
                                x2, y2, x3, y3 = xf_3, yf_3, x_3, y_3
                                d2 = d3 = dis_3m
                                conti == 2
                            else:
                                cx, cy = (x_3 + xf_3) / 2, (y_3 + yf_3) / 2
                                x__2, y__2 = cx + (x_3 - x4), cy + (y_3 - y4)
                                x__4, y__4 = cx + (xf_2 - xf_3), cy + (yf_2 - yf_3)
                                c_x, c_y = (x__2 + x__4) / 2, (y__2 + y__4) / 2
                                xf__2, yf__2 = findIntersection(x_2, y_2, xf_2, yf_2, cx, cy, c_x, c_y)
                                if xf__2:
                                    xf_2, yf_2 = xf__2, yf__2;
                                    dis_2m = math.sqrt((xf_2 - x_2) ** 2 + (yf_2 - y_2) ** 2)
                                    no_feature[-2] = [xf_2, yf_2]
                                    no_feature.pop(2)
                                    no_feature.pop(-3)
                                    if dis_2m <= 3.5:
                                        x2, y2, x3, y3 = x_2, y_2, xf_2, yf_2
                                        d2 = d3 = dis_2m
                                        conti = 2

        if conti:
            x2, y2 = no_feature[1]
            x3, y3 = no_feature[-2]
            A1, B1, C1 = two_points_line(x1, y1, x2, y2, x1, y1)
            A3, B3, C3 = two_points_line(x3, y3, x4, y4, x3, y3)
            angle_23 = cal_angle((x3 - x2, y3 - y2), vec_se)
            min23 = min(angle_23 - 0, 90 - angle_23)
            if conti == 1:  # cx两边投影
                if min23 >= (min12 + min34) / 2:
                    cx, cy = (x2 + x3) / 2, (y2 + y3) / 2
                    x_2, y_2 = Drop_point(A1, B1, C1, cx, cy)
                    x_3, y_3 = Drop_point(A3, B3, C3, cx, cy)
                    if angle_13 <= 5:
                        area_m = polygon_area([[cx, cy], [x2, y2], [x_2, y_2]]) + polygon_area(
                            [[cx, cy], [x3, y3], [x_3, y_3]])
                        if area_judg(area_m, contour_area, 2, seg_type):
                            no_feature[1:-1] = [x_2, y_2], [x_3, y_3]
                    elif min12 <= min34:
                        if area_judg(polygon_area([[x3, y3], [x2, y2], [x_2, y_2]]), contour_area, 2, seg_type):
                            no_feature[1] = [x_2, y_2]
                    elif area_judg(polygon_area([[x3, y3], [x_3, y_3], [x2, y2]]), contour_area, 2, seg_type):
                        no_feature[-2] = [x_3, y_3]
            elif conti == 2:  # 拟合为一条直线
                cx, cy = (x2 + x3) / 2, (y2 + y3) / 2
                x_2, y_2 = cx + (x2 - x1), cy + (y2 - y1)
                x_4, y_4 = cx + (x4 - x3), cy + (y4 - y3)
                c_x, c_y = (x_2 + x_4) / 2, (y_2 + y_4) / 2
                Ac, Bc, Cc = two_points_line(cx, cy, c_x, c_y, cx, cy)
                x_1, y_1 = Drop_point(Ac, Bc, Cc, x1, y1)
                x_4, y_4 = Drop_point(Ac, Bc, Cc, x4, y4)
                area_m = math.sqrt((y_4 - y_1) ** 2 + (x_4 - x_1) ** 2) * (d2 + d3) / 4
                if area_judg(area_m, contour_area, 2, seg_type):
                    no_feature = [[x_1, y_1], [x_4, y_4]]
            elif conti == 3:  # 中点两边投影
                if d2 + d3 <= 8:
                    no_feature.pop(2)
                else:
                    cx2, cy2, cx3, cy3 = (x2 + midx) / 2, (y2 + midy) / 2, (x3 + midx) / 2, (y3 + midy) / 2
                    x_2, y_2 = findIntersection(x1, y1, x2, y2, cx2, cy2, cx3, cy3)
                    x_3, y_3 = findIntersection(x3, y3, x4, y4, cx2, cy2, cx3, cy3)
                    if x_2 and x_3 and min(abs(cal_ang((x1, y1), (x_2, y_2), (x_3, y_3)) - 90),
                                           abs(cal_ang((x_2, y_2), (x_3, y_3), (x4, y4)) - 90)) <= 15:
                        area_m = union([[x2, y2], [x3, y3], [x_3, y_3], [x_2, y_2]], [[x2, y2], [midx, midy], [x3, y3]])
                        if area_judg(area_m, contour_area, 2, seg_type):
                            no_feature[1:-1] = [x_2, y_2], [x_3, y_3]
                    else:
                        x_2, y_2 = Drop_point(A1, B1, C1, midx, midy)
                        x_3, y_3 = Drop_point(A3, B3, C3, midx, midy)
                        area_m = polygon_area([[x2, y2], [midx, midy], [x_2, y_2]])
                        if abs(90 - angle2) > 10 and area_judg(area_m, contour_area, 1, seg_type) and min2m >= min12:
                            no_feature[1] = [x_2, y_2]
                            conti += 1
                        area_m = polygon_area([[x3, y3], [midx, midy], [x_3, y_3]])
                        if abs(90 - angle3) > 10 and area_judg(area_m, contour_area, 1, seg_type) and min3m >= min34:
                            no_feature[-2] = [x_3, y_3]
                            conti += 1
                        if conti == 5:
                            no_feature.pop(2)
            elif conti == 4:  # 锐角扩展形式
                Ac, Bc, Cc = two_points_line(x1, y1, x2, y2, midx, midy) if min12 <= min34 else two_points_line(x3, y3,
                                                                                                                x4, y4,
                                                                                                                midx,
                                                                                                                midy)
                cx_2m, cy_2m, cx_3m, cy_3m = x2 + (midx - x2) / 4, y2 + (midy - y2) / 4, x3 + (midx - x3) / 4, y3 + (
                        midy - y3) / 4
                x_2, y_2 = Drop_point(A1, B1, C1, cx_2m, cy_2m)
                midx_l, midy_l = Drop_point(Ac, Bc, Cc, cx_2m, cy_2m)
                x_3, y_3 = Drop_point(A3, B3, C3, cx_3m, cy_3m)
                midx_r, midy_r = Drop_point(Ac, Bc, Cc, cx_3m, cy_3m)
                area_c_2m = polygon_area([[x2, y2], [x_2, y_2], [cx_2m, cy_2m]]) + polygon_area(
                    [[midx, midy], [midx_l, midy_l], [cx_2m, cy_2m]])
                area_c_3m = polygon_area([[x3, y3], [x_3, y_3], [cx_3m, cy_3m]]) + polygon_area(
                    [[midx, midy], [midx_r, midy_r], [cx_3m, cy_3m]])
                if area_judg(area_c_2m, contour_area, 1, seg_type):
                    no_feature[1] = [x_2, y_2]
                    no_feature[2] = [midx_l, midy_l]
                if area_judg(area_c_3m, contour_area, 1, seg_type):
                    no_feature[-2] = [x_3, y_3]
                    no_feature.insert(3, [midx_r, midy_r])


    elif angle_13 >= 170:
        conti = False
        if 3 < d2 <= 40:
            area_1 = polygon_area(no_feature[1:-1])
            if total == 4:
                conti = 1
            elif total == 5:
                conti = 3 if abs(angle2 - 90) <= 15 and angle3 <= 60 else 4 if abs(
                    angle3 - 90) <= 15 and angle2 <= 60 else 2
            else:
                dis_2f = math.sqrt((y2 - yf2) ** 2 + (x2 - xf2) ** 2)
                dis_3f = math.sqrt((y3 - yf3) ** 2 + (x3 - xf3) ** 2)
                if rayCasting([x2, y2], app2) == rayCasting([x3, y3], app3) == rayCasting(no_feature[2],
                                                                                          appf2) == rayCasting(
                    no_feature[3], appf3) and rayCasting9(no_feature[2], appf2) != None and rayCasting9(
                    no_feature[3], appf3) != None and dis_12 + dis_34 >= dis_2f + dis_3f \
                        and max(df2_1, df2_3, df3_1, df3_3) <= (d2 + d3) / 2:
                    # show_result(approx,approx,x1,y1,True)
                    conti = 2

            if conti == 2:
                c_x, c_y = (x1 + x4) / 2, (y1 + y4) / 2
                Ac, Bc, Cc = two_points_vline(x1, y1, x2, y2, c_x, c_y) if min12 <= min34 else two_points_vline(x3, y3,
                                                                                                                x4, y4,
                                                                                                                c_x,
                                                                                                                c_y)
                df = du = dw = 0
                up = down = None
                for i in no_feature[1:-1]:
                    di = abs(Ac * i[0] + Bc * i[1] + Cc) / math.sqrt(Ac * Ac + Bc * Bc)
                    if di > df:
                        far = i
                        df = di
                    d_f1 = abs(A1 * i[0] + B1 * i[1] + C1) / math.sqrt(A1 * A1 + B1 * B1)
                    d_f3 = abs(A3 * i[0] + B3 * i[1] + C3) / math.sqrt(A3 * A3 + B3 * B3)
                    if d_f3 >= max(d_f1, d2, du):
                        up = i
                        du = d_f3
                    elif d_f1 >= max(d_f3, d3, dw):
                        down = i
                        dw = d_f1
                if total == 5:
                    for i in range(len(no_feature[2:-2]) - 1, -1, -1):
                        if no_feature[i + 2] != up and no_feature[i + 2] != down and no_feature[
                            i + 2] != far and rayCasting([x2, y2], app2) != rayCasting([midx, midy], appf2):
                            conti -= 1

            if conti == 1:
                ang_1 = cal_ang((x2, y2), (x3, y3), (x4, y4))
                if ang_1 < 80 or ang_1 > 100:
                    c_x, c_y = (x2 + x3) / 2, (y2 + y3) / 2
                    x_2, y_2 = Drop_point(A1, B1, C1, c_x, c_y)
                    x_3, y_3 = Drop_point(A3, B3, C3, c_x, c_y)
                    area_1 = union(no_feature, [[x1, y1], [x_2, y_2], [x_3, y_3], [x4, y4]])
                    if (x_2 - x1) * (x2 - x1) >= 0 and (x_3 - x4) * (x3 - x4) >= 0 and area_judg(area_1, contour_area,
                                                                                                 2,
                                                                                                 seg_type) and dis_23 < 20:
                        no_feature[1:-1] = [x_2, y_2], [x_3, y_3]
            elif conti == 2:
                # no_featur=np.array(no_feature)
                # no_featur[:,0],no_featur[:,1]=no_featur[:,0]-minx+10,no_featur[:,1]-miny+10
                # print(no_featur)
                Au, Bu, Cu = two_points_line(x1, y1, x2, y2, up[0], up[1])
                if up != no_feature[1] and min2m >= min12:
                    index_u = no_feature.index(up)
                    if cal_ang([x1, y1], [x2, y2], up) < 90:
                        x_2, y_2 = Drop_point(A1, B1, C1, up[0], up[1])
                        area_u = polygon_area([up, [x_2, y_2], [x2, y2]])
                        if area_judg(area_u, contour_area, 2, seg_type):
                            no_feature[1] = [x_2, y_2]
                        else:
                            cx, cy = up[0] + (x2 - up[0]) / 5 * 2, up[1] + (y2 - up[1]) / 5 * 2
                            x_2, y_2 = Drop_point(A1, B1, C1, cx, cy)
                            up_nx, up_ny = Drop_point(Au, Bu, Cu, cx, cy)
                            no_feature[1] = [x_2, y_2]
                            no_feature[index_u] = [up_nx, up_ny]
                            up = [up_nx, up_ny]
                    elif cal_ang([x1, y1], [x2, y2], up) > 90:
                        x_2, y_2 = Drop_point(Au, Bu, Cu, x2, y2)
                        area_u = polygon_area([[x2, y2], [x_2, y_2], up])
                        if area_judg(area_u, contour_area, 2, seg_type):
                            no_feature.insert(2, [x_2, y_2])
                        else:
                            cx, cy = x2 + (up[0] - x2) / 5 * 2, y2 + (up[1] - y2) / 5 * 2
                            x_2, y_2 = Drop_point(A1, B1, C1, cx, cy)
                            up_nx, up_ny = Drop_point(Au, Bu, Cu, cx, cy)
                            no_feature[1] = [x_2, y_2]
                            no_feature.insert(2, [up_nx, up_ny])
                if up != far:
                    far_ux, far_uy = Drop_point(Au, Bu, Cu, far[0], far[1])
                    if total == 5:
                        minm = min2m if far == [midx, midy] else min3m
                        index_u = no_feature.index(up)
                        area_u = polygon_area([far, [far_ux, far_uy], up])
                        if minm >= min12 and area_judg(area_u, contour_area, 4, seg_type):
                            if up != no_feature[1] and cal_ang([x1, y1], [x2, y2], up) <= 90:
                                no_feature.insert(index_u + 1, [far_ux, far_uy])
                            else:
                                no_feature[index_u] = [far_ux, far_uy]
                    else:
                        index_f = no_feature.index(far)
                        area_u = no_feature[1:index_f + 1]
                        area_u.append([far_ux, far_uy])
                        area_u = polygon_area(area_u)
                        if area_judg(area_u, contour_area, 4, seg_type):
                            no_feature[1:index_f] = [[far_ux, far_uy]]

                Ad, Bd, Cd = two_points_line(x3, y3, x4, y4, down[0], down[1])
                if down != no_feature[-2] and min3m >= min34:
                    index_d = no_feature.index(down)
                    if cal_ang([x4, y4], [x3, y3], down) < 90:
                        x_3, y_3 = Drop_point(A3, B3, C3, down[0], down[1])
                        area_d = polygon_area([down, [x_3, y_3], [x3, y3]])
                        if area_judg(area_d, contour_area, 2, seg_type):
                            no_feature[-2] = [x_3, y_3]
                        else:
                            cx, cy = down[0] + (x3 - down[0]) / 5 * 2, down[1] + (y3 - down[1]) / 5 * 2
                            x_3, y_3 = Drop_point(A3, B3, C3, cx, cy)
                            down_nx, down_ny = Drop_point(Ad, Bd, Cd, cx, cy)
                            no_feature[-2] = [x_3, y_3]
                            no_feature[index_d] = [down_nx, down_ny]
                            down = [down_nx, down_ny]
                    elif cal_ang([x4, y4], [x3, y3], down) > 90:
                        x_3, y_3 = Drop_point(Ad, Bd, Cd, x3, y3)
                        area_d = polygon_area([[x3, y3], [x_3, y_3], down])
                        if area_judg(area_d, contour_area, 2, seg_type):
                            no_feature.insert(-2, [x_3, y_3])
                        else:
                            cx, cy = x3 + (down[0] - x3) / 5 * 2, y3 + (down[1] - y3) / 5 * 2
                            x_3, y_3 = Drop_point(A3, B3, C3, cx, cy)
                            down_nx, down_ny = Drop_point(Ad, Bd, Cd, cx, cy)
                            no_feature[-2] = [x_3, y_3]
                            no_feature.insert(-2, [down_nx, down_ny])
                if down != far:
                    far_dx, far_dy = Drop_point(Ad, Bd, Cd, far[0], far[1])
                    if total == 5:
                        minm = min3m if far == [midx, midy] else min2m
                        index_d = no_feature.index(down)
                        area_d = polygon_area([far, [far_dx, far_dy], down])
                        if minm >= min34 and area_judg(area_d, contour_area, 4, seg_type):
                            if down != no_feature[-2] and cal_ang([x4, y4], [x3, y3], down) <= 90:
                                no_feature.insert(index_d, [far_dx, far_dy])
                            else:
                                no_feature[index_d] = [far_dx, far_dy]
                    elif total == 6:
                        index_f = no_feature.index(far)
                        area_d = no_feature[index_f:-1]
                        area_d.append([far_dx, far_dy])
                        area_d = polygon_area(area_d)
                        if area_judg(area_d, contour_area, 4, seg_type):
                            no_feature[index_f + 1:-1] = [[far_dx, far_dy]]
            elif conti == 3:
                Ac, Bc, Cc = two_points_line(x3, y3, x4, y4, midx, midy)
                x_3, y_3 = Drop_point(Ac, Bc, Cc, x3, y3)
                no_feature.insert(3, [x_3, y_3])
            elif conti == 4:
                Ac, Bc, Cc = two_points_line(x1, y1, x2, y2, midx, midy)
                x_2, y_2 = Drop_point(Ac, Bc, Cc, x2, y2)
                no_feature.insert(2, [x_2, y_2])
        elif d2 <= 3 and polygon_area(no_feature) <= 25:
            del no_feature[1:-1]

    elif 83 < angle_13 <= 97:
        if max(d2, d3) > 25:
            return no_feature, show, no_feature[0][0], no_feature[0][1]
        x_23, y_23 = findIntersection(x1, y1, x2, y2, x3, y3, x4, y4)
        area = no_feature[1:-1]
        area_0 = polygon_area(area)
        area.append([x_23, y_23])
        area_1 = polygon_area(area)
        conti = False

        condition1, condition2, condition3 = total == 4 and angle2 > 150 and min1m <= min(min12,
                                                                                          min23) and min12x > 4, total == 4 and angle3 > 150 and min4m <= min(
            min34, min23) and min34x > 4, total == 4 and min23 >= max(min12, min34)
        if rayCasting9([x_23, y_23], approx) == False and area_judg(max(area_0, area_1), contour_area, 1,
                                                                    seg_type) and (
                condition1 or condition2 or condition3 or total == 5 and not max(abs(90 - angle2), abs(90 - anglem),
                                                                                 abs(90 - angle3)) <= 20):
            if condition1:
                cx_01, cy_01 = (x1 + x2) / 2, (y1 + y2) / 2;
                cx_12, cy_12 = (x2 + x3) / 2, (y2 + y3) / 2
                Ac, Bc, Cc = two_points_line(cx_01, cy_01, cx_12, cy_12, cx_01, cy_01)
                x_1, y_1 = Drop_point(Ac, Bc, Cc, x1, y1);
                x_3, y_3 = Drop_point(Ac, Bc, Cc, x3, y3)
                area_m = union(no_feature[:3], [[x1, y1], [x3, y3], [x_3, y_3], [x_1, y_1]])
                if area_judg(area_m, contour_area, 1, seg_type):
                    no_feature[0] = [x_1, y_1];
                    no_feature[2] = [x_3, y_3];
                    no_feature.pop(1)
            elif condition2:
                cx_01, cy_01 = (x4 + x3) / 2, (y4 + y3) / 2;
                cx_12, cy_12 = (x2 + x3) / 2, (y2 + y3) / 2
                Ac, Bc, Cc = two_points_line(cx_01, cy_01, cx_12, cy_12, cx_01, cy_01)
                x_4, y_4 = Drop_point(Ac, Bc, Cc, x4, y4);
                x_2, y_2 = Drop_point(Ac, Bc, Cc, x2, y2)
                area_m = union(no_feature[1:], [[x4, y4], [x2, y2], [x_2, y_2], [x_4, y_4]])
                if area_judg(area_m, contour_area, 1, seg_type):
                    no_feature[1] = [x_2, y_2];
                    no_feature[3] = [x_4, y_4];
                    no_feature.pop(-2)
            else:
                no_feature[1:-1] = [[x_23, y_23]]

        elif total == 4:
            if angle2 > 90 and angle3 > 90:
                if area_judg(area_1, contour_area, 2, seg_type) and min23 >= max(min12, min34) and dis_23 < 20:
                    no_feature[1:-1] = [[x_23, y_23]]
            elif angle2 < 90 and angle3 > 90:
                angle_2, area_3 = cal_ang((x1, y1), (x2, y2), (x4, y4)), polygon_area([[x2, y2], [x3, y3], [x4, y4]])
                angle_24 = cal_angle((x4 - x2, y4 - y2), vec_se);
                min24 = min(90 - angle_24, angle_24 - 0)
                intersect_12 = lineup(x1, y1, x2, y2, x_23, y_23)
                if angle2 > 75 and rayCasting([x3, y3], app3) and (
                        abs(angle_2 - 90) <= abs(angle_13 - 90) and min23 > min34 or min24 <= min34) and area_judg(
                    area_3, contour_area, 2, seg_type):
                    no_feature.pop(-2)
                elif rayCasting([x2, y2], app2) and intersect_12 and area_judg(area_1, contour_area, 2,
                                                                               seg_type) and min23 >= max(min12, min34):
                    no_feature[1:-1] = [[x_23, y_23]]
                elif area_judg(area_1, contour_area, -4, seg_type):
                    no_feature[1:-1] = [[x_23, y_23]]
                else:
                    conti = 1
            elif angle2 > 90 and angle3 < 90:
                angle_3, area_2 = cal_ang((x1, y1), (x3, y3), (x4, y4)), polygon_area([[x1, y1], [x2, y2], [x3, y3]])
                angle_31 = cal_angle((x3 - x1, y3 - y1), vec_se);
                min31 = min(90 - angle_31, angle_31 - 0)
                intersect_34 = min(x3, x4) <= x_23 <= max(x3, x4) and min(y3, y4) <= y_23 <= max(y3, y4)
                if angle3 > 75 and rayCasting([x2, y2], app2) and (
                        abs(angle_3 - 90) <= abs(angle_13 - 90) and min23 > min12 or min31 <= min12) and area_judg(
                    area_2, contour_area, 2, seg_type):
                    no_feature.pop(1)
                elif rayCasting([x3, y3], app3) and intersect_34 and area_judg(area_1, contour_area, 2,
                                                                               seg_type) and min23 >= max(min12, min34):
                    no_feature[1:-1] = [[x_23, y_23]]
                elif area_judg(area_1, contour_area, -4, seg_type):
                    no_feature[1:-1] = [[x_23, y_23]]
                else:
                    conti = 1
            elif angle2 < 90 and angle3 < 90:
                conti = 1
            if conti == 1:
                if min34 <= min12:
                    A32, B32, C32 = two_points_line(x3, y3, x4, y4, x2, y2)
                    x_3, y_3 = Drop_point(A32, B32, C32, x3, y3)
                    no_feature[2:-2] = [[x_3, y_3]]
                else:
                    A32, B32, C32 = two_points_line(x1, y1, x2, y2, x3, y3)
                    x_2, y_2 = Drop_point(A32, B32, C32, x2, y2)
                    no_feature[2:-2] = [[x_2, y_2]]
        elif total == 5:
            conti = False
            if rayCasting([x2, y2], app2) == rayCasting([x3, y3], app3) == rayCasting([midx, midy], appf2):
                conti = 1 if abs(360 - (angle2 + anglem + angle3 + 180 - angle_13)) <= 7 else -1
            elif rayCasting([x2, y2], app2) == rayCasting([x3, y3], app3) != rayCasting([midx, midy], appf2):
                conti = 1 if abs(360 - (angle2 + anglem + angle3 + 180 - angle_13)) <= 7 else 2
            elif rayCasting([x2, y2], app2) != rayCasting([x3, y3], app3):
                conti = 3
            if conti == -1:
                if area_judg(area_1, contour_area, 2, seg_type) and min2m >= min12 and min3m >= min34:
                    no_feature[1:-1] = [[x_23, y_23]]
            elif conti == 1:
                dis_f2, dis_f3 = abs(A1 * midx + B1 * midy + C1) / math.sqrt(A1 * A1 + B1 * B1), abs(
                    A3 * midx + B3 * midy + C3) / math.sqrt(A3 * A3 + B3 * B3)
                up = [x3, y3] if d3 > dis_f2 else [midx, midy]
                far = [x2, y2] if d2 > dis_f3 else [midx, midy]
                x_2, y_2 = Drop_point(A1, B1, C1, far[0], far[1])
                x_3, y_3 = Drop_point(A3, B3, C3, up[0], up[1])
                if min12 <= min34:
                    A_1, B_1, C_1 = two_points_line(x1, y1, x2, y2, up[0], up[1])
                    mid_x, mid_y = Drop_point(A_1, B_1, C_1, far[0], far[1])
                else:
                    A_1, B_1, C_1 = two_points_line(x3, y3, x4, y4, far[0], far[1])
                    mid_x, mid_y = Drop_point(A_1, B_1, C_1, up[0], up[1])
                area_2, area_3 = polygon_area([far, [x2, y2], [x_2, y_2]]), polygon_area([up, [x3, y3], [x_3, y_3]])
                if area_judg(area_2, contour_area, 1, seg_type) and min2m >= min12:
                    no_feature[1] = [x_2, y_2]
                if area_judg(area_3, contour_area, 1, seg_type) and min3m >= min34:
                    no_feature[-2] = [x_3, y_3]
                area_f = union(no_feature, [[x1, y1], [x_2, y_2], [mid_x, mid_y], [x_3, y_3], [x4, y4]])
                if area_judg(area_f, contour_area, 1, seg_type):
                    no_feature[2] = [mid_x, mid_y]
            elif conti == 2 or conti == 3:
                if intersect_2m or intersect_3m:
                    area_f = area_c_2m + polygon_area(
                        [[c_x_2m, c_y_2m], [x_23, y_23], [x2, y2]]) if intersect_2m else area_c_3m + polygon_area(
                        [[c_x_3m, c_y_3m], [x_23, y_23], [x3, y3]])
                    if area_judg(area_f, contour_area, 1, seg_type) and rayCasting([midx, midy],
                                                                                   appf2) and min34 < min3m if intersect_2m else min12 < min2m:
                        no_feature[1:-1] = [[x_23, y_23]]
                else:
                    if conti == 2:
                        cx_2m, cy_2m, cx_3m, cy_3m = (midx + x2) / 2, (midy + y2) / 2, (midx + x3) / 2, (midy + y3) / 2
                    elif rayCasting([x3, y3], app3) != rayCasting([midx, midy], appf2):
                        cx_2m, cy_2m, cx_3m, cy_3m = x2 + (midx - x2) / 3 * 2, y2 + (midy - y2) / 3 * 2, x3 + (
                                midx - x3) / 3, y3 + (midy - y3) / 3
                    else:
                        cx_2m, cy_2m, cx_3m, cy_3m = x2 + (midx - x2) / 3, y2 + (midy - y2) / 3, x3 + (
                                midx - x3) / 3 * 2, y3 + (midy - y3) / 3 * 2
                    x_2, y_2 = Drop_point(A1, B1, C1, cx_2m, cy_2m)
                    x_3, y_3 = Drop_point(A3, B3, C3, cx_3m, cy_3m)
                    if min12 <= min34:
                        Ac, Bc, Cc = two_points_vline(x1, y1, x2, y2, x_2, y_2)
                        mid_x, mid_y = Drop_point(Ac, Bc, Cc, x_3, y_3)
                    else:
                        Ac, Bc, Cc = two_points_vline(x3, y3, x4, y4, x_3, y_3)
                        mid_x, mid_y = Drop_point(Ac, Bc, Cc, x_2, y_2)
                    area_f = union([[x2, y2], [midx, midy], [x3, y3], [x_23, y_23]],
                                   [[x_2, y_2], [mid_x, mid_y], [x_3, y_3], [x_23, y_23]])
                    if area_judg(area_f, contour_area, 2, seg_type) and min2m >= min12 and min3m >= min34:
                        no_feature[1:-1] = [x_2, y_2], [mid_x, mid_y], [x_3, y_3]
        elif total == 6:
            if rayCasting([x2, y2], app2) != rayCasting([x3, y3], app3) and rayCasting(no_feature[2],
                                                                                       appf2) == rayCasting(
                no_feature[3], appf3) and min(cal_angle((xf2 - x2, yf2 - y2), (x3 - xf3, y3 - yf3)),
                                              abs(90 - cal_ang((x3, y3), (xf3, yf3), (xf2, yf2))),
                                              abs(90 - cal_ang((x2, y2), (xf2, yf2), (xf3, yf3)))) > 10:
                if rayCasting([x2, y2], app2) != rayCasting(no_feature[2], appf2) and angle2 >= 70:
                    conti = 2
                elif rayCasting([x3, y3], app3) != rayCasting(no_feature[2], appf2) and angle3 >= 70:
                    no_feature = no_feature[::-1]
                    conti = 3
                if conti == 2 or conti == 3:
                    x1, y1 = no_feature[0];
                    x2, y2 = no_feature[1];
                    xf2, yf2 = no_feature[2];
                    xf3, yf3 = no_feature[3];
                    x3, y3 = no_feature[-2];
                    x4, y4 = no_feature[-1]
                    dis_f2, dis_f3 = abs(A1 * xf2 + B1 * yf2 + C1) / math.sqrt(A1 * A1 + B1 * B1), abs(
                        A1 * xf3 + B1 * yf3 + C1) / math.sqrt(A1 * A1 + B1 * B1)
                    far = no_feature[2] if dis_f2 >= dis_f3 else no_feature[-3]
                    Af, Bf, Cf = two_points_line(startx, starty, endx, endy, far[0], far[1]) if cal_angle(
                        (x2 - x1, y2 - y1), vec_se) <= 20 else two_points_vline(startx, starty, endx, endy, far[0],
                                                                                far[1])
                    x_2, y_2 = Drop_point(Af, Bf, Cf, x2, y2)
                    x_3, y_3 = Drop_point(Af, Bf, Cf, x3, y3)
                    index_f = no_feature.index(far)
                    area_d, area_u = no_feature[1:index_f + 1], no_feature[index_f:-1]
                    area_d.append([x_2, y_2])
                    area_u.append([x_3, y_3])
                    area_d, area_u = polygon_area(area_d), polygon_area(area_u)
                    if area_judg(area_d, contour_area, 4, seg_type):
                        no_feature[2:index_f] = [[x_2, y_2]]
                    if area_judg(area_u, contour_area, 4, seg_type):
                        no_feature[index_f + 1:-1] = [[x_3, y_3]]
                    if conti == 3:
                        no_feature = no_feature[::-1]

    return no_feature, show, x1, y1


def show_result(appro, approx, x1, y1, show):
    return


def Drop_point(A, B, C, x, y):
    x_0 = (B * B * x - A * B * y - A * C) / (A * A + B * B)
    y_0 = (A * A * y - A * B * x - B * C) / (A * A + B * B)
    return x_0, y_0


def two_points_line(x0, y0, x1, y1, x3, y3):
    if x1 - x0 == 0:
        A = -1
        B = 0
        C = x3
    else:
        A = (y1 - y0) / (x1 - x0)
        B = -1
        C = y3 - A * x3
    return A, B, C


def cal_ang(point_1, point_2, point_3):
    a = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]))
    b = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1]))
    c = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))
    d = (b * b - a * a - c * c) / (-2 * a * c)
    if d < -1:
        d = -1
    elif d > 1:
        d = 1
    B = math.degrees(math.acos(d))  # 将弧度转为角度
    return B


def cal_angle(v1, v2, half=True):
    angle1 = math.atan2(v1[1], v1[0])
    angle1 = int(angle1 * 180 / math.pi)
    angle2 = math.atan2(v2[1], v2[0])
    angle2 = int(angle2 * 180 / math.pi)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    if half == True and included_angle > 90:
        included_angle = 180 - included_angle
    return included_angle


def coss_multi(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def polygon_area(polygon):
    n = len(polygon)
    polygon = np.array(polygon)
    if n < 3:
        return 0
    vectors = np.zeros((n, 2))
    for i in range(0, n):
        vectors[i, :] = polygon[i, :] - polygon[0, :]
    area = 0
    for i in range(1, n):
        area = area + coss_multi(vectors[i - 1, :], vectors[i, :]) / 2
    return abs(area)


def union(a, b):
    poly1 = Polygon(np.array(a)).convex_hull
    poly2 = Polygon(np.array(b)).convex_hull
    try:
        inter_area = poly1.intersection(poly2).area  # 相交面积
        union_area = poly1.area + poly2.area - 2 * inter_area
        return union_area
    except shapely.geos.TopologicalError:
        return False


def two_points_line(x0, y0, x1, y1, x3, y3):
    if x1 - x0 == 0:
        A = -1
        B = 0
        C = x3
    else:
        A = (y1 - y0) / (x1 - x0)
        B = -1
        C = y3 - A * x3
    return A, B, C


def two_points_vline(x0, y0, x1, y1, x3, y3):
    if y1 - y0 == 0:
        A = -1
        B = 0
        C = x3
    elif x1 - x0 == 0:
        A = 0
        B = -1
        C = y3
    else:
        A = -1 / ((y1 - y0) / (x1 - x0))
        B = -1
        C = y3 - A * x3
    return A, B, C


def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    if (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) == 0:
        return None, None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return px, py


def lineup(x1, y1, x2, y2, x, y, equal=True):
    if equal:
        return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)
    else:
        return min(x1, x2) < x < max(x1, x2) and min(y1, y2) < y < max(y1, y2)


def rayCasting(p, poly):
    px = p[0]
    py = p[1]
    flag = False
    i = 0
    l = len(poly)
    j = l - 1
    while i < l:
        sx = poly[i][0]
        sy = poly[i][1]
        tx = poly[j][0]
        ty = poly[j][1]
        if (sx == px and sy == py) or (tx == px and ty == py):
            return True
        if (sy < py and ty >= py) or (sy >= py and ty < py):
            x = sx + (py - sy) * (tx - sx) / (ty - sy)
            if x == px:
                return True
            if x > px:
                flag = not flag
        j = i
        i += 1
    return True if flag else False


def rayCasting9(p, poly):
    a = []
    for m in range(-1, 2):
        for n in range(-1, 2):
            q = [p[0] + m, p[1] + n]
            a.append(rayCasting(q, poly))
    return True if set(a) == {True} else False if set(a) == {False} else None


def horv(approx, i, vec_02, dis_02, l1, yan):
    h, v = False, False
    for j in range(len(approx) - 1, -1, -1):
        if j != i:
            x3, y3 = approx[j]
            x4, y4 = approx[j - 1]
            vec_34 = (x4 - x3, y4 - y3)
            dis_34 = math.sqrt((y4 - y3) ** 2 + (x4 - x3) ** 2)
            dis = (dis_02 + dis_34) / 2
            angle_vh = cal_angle(vec_02, vec_34)
            if not yan:
                if angle_vh <= 5 and (dis_34 > 20 and dis > l1 or dis_34 >= 25):
                    h = True
                elif angle_vh >= 85 and (dis_34 > 20 and dis > l1 or dis_34 >= 25):
                    v = True
            else:
                if angle_vh <= 5 and dis_02 + dis_34 >= 40:  # and dis_34>l1
                    h = True
                elif angle_vh >= 85 and dis_02 + dis_34 >= 40:  # and dis_34>l1
                    v = True
    return h, v


def betterh(approx, i, vec_01, vec_02, minx, miny, jia):
    better = True
    m = i + 4 if jia else i + 3
    for j in range(m, m - 7, -1):
        if j not in [i + 2, i + 1, i, i - 1]:
            x1, y1 = approx[j % len(approx)]
            x2, y2 = approx[(j - 1) % len(approx)]
            vec_12, dis_12 = (x2 - x1, y2 - y1), math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            angle_02, angle_01 = cal_angle(vec_02, vec_12), cal_angle(vec_01, vec_12)
            if angle_01 <= 5 and dis_12 > 25 and angle_02 > angle_01 + 1:
                better = False
    return better


####################################################################################主方向
def zhu(approx, minx, miny, horv, seg_type):
    lenth = 0
    total = len(approx)
    storge_vh, storge_h = [], []
    app = approx.copy()
    if seg_type == 'bfp' and total <= 100:
        for i in range(total - 1, -1, -1):
            x1, y1 = app[i]
            x2, y2 = app[i - 1]
            vec_12 = (x2 - x1, y2 - y1)
            angle_h = cal_angle(vec_12, (1, 0))
            dis_12 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dis_12 > lenth:
                startx, starty, endx, endy, lenth = x1, y1, x2, y2, dis_12
            if min(angle_h - 0, 90 - angle_h) < 7:
                storge_h.append([x1, y1, x2, y2, dis_12])
            for j in range(i - 1, -1, -1):
                x3, y3 = app[j]
                x4, y4 = app[j - 1]
                vec_34 = (x4 - x3, y4 - y3)
                angle_vh = cal_angle(vec_12, vec_34)
                if angle_vh <= 5 or angle_vh >= 85:
                    storge_vh.append([x1, y1, x2, y2])
                    storge_vh.append([x3, y3, x4, y4])
        if horv:
            lenth3 = 0.95 * lenth
            for i in storge_h:
                if i[4] >= lenth3:
                    startx, starty, endx, endy, lenth3 = i
                    vec_se = (endx - startx, endy - starty)
                    return startx, starty, endx, endy, vec_se
        lenth2 = 0.7 * lenth
        for i in range(0, len(storge_vh), 2):
            dis_12 = math.sqrt((storge_vh[i][2] - storge_vh[i][0]) ** 2 + (storge_vh[i][3] - storge_vh[i][1]) ** 2)
            dis_34 = math.sqrt(
                (storge_vh[i + 1][2] - storge_vh[i + 1][0]) ** 2 + (storge_vh[i + 1][3] - storge_vh[i + 1][1]) ** 2)
            p_12 = dis_12 / (dis_12 + dis_34)
            dis_14 = dis_12 * p_12 + dis_34 * (1 - p_12)
            if dis_14 > lenth2:
                if dis_12 >= dis_34:
                    startx, starty, endx, endy = storge_vh[i][0], storge_vh[i][1], storge_vh[i][2], storge_vh[i][3]
                else:
                    startx, starty, endx, endy = storge_vh[i + 1][0], storge_vh[i + 1][1], storge_vh[i + 1][2], \
                                                 storge_vh[i + 1][3]
                lenth2 = dis_14
        if horv:
            lenth4 = 0.9 * lenth2
            for i in range(0, len(storge_vh), 2):
                angle_h = cal_angle((storge_vh[i][2] - storge_vh[i][0], storge_vh[i][3] - storge_vh[i][1]), (1, 0))
                angle_v = cal_angle(
                    (storge_vh[i + 1][2] - storge_vh[i + 1][0], storge_vh[i + 1][3] - storge_vh[i + 1][1]), (1, 0))
                dis_12 = math.sqrt((storge_vh[i][2] - storge_vh[i][0]) ** 2 + (storge_vh[i][3] - storge_vh[i][1]) ** 2)
                dis_34 = math.sqrt(
                    (storge_vh[i + 1][2] - storge_vh[i + 1][0]) ** 2 + (storge_vh[i + 1][3] - storge_vh[i + 1][1]) ** 2)
                p_12 = dis_12 / (dis_12 + dis_34)
                dis_14 = dis_12 * p_12 + dis_34 * (1 - p_12)
                if min(angle_h - 0, 90 - angle_h) + min(angle_v - 0, 90 - angle_v) <= 10 and dis_14 > lenth4:
                    lenth4 = dis_14
                    if dis_12 >= dis_34:
                        startx, starty, endx, endy = storge_vh[i][0], storge_vh[i][1], storge_vh[i][2], storge_vh[i][3]
                    else:
                        startx, starty, endx, endy = storge_vh[i + 1][0], storge_vh[i + 1][1], storge_vh[i + 1][2], \
                                                     storge_vh[i + 1][3]
    else:
        for i in range(total - 1, -1, -1):
            x1, y1 = app[i]
            x2, y2 = app[i - 1]
            vec_12 = (x2 - x1, y2 - y1)
            angle_h = cal_angle(vec_12, (1, 0))
            dis_12 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dis_12 > lenth:
                startx, starty, endx, endy, lenth = x1, y1, x2, y2, dis_12
            if min(angle_h - 0, 90 - angle_h) < 7:
                storge_h.append([x1, y1, x2, y2, dis_12])
        if horv:
            lenth3 = 0.85 * lenth
            for i in storge_h:
                if i[4] >= lenth3:
                    startx, starty, endx, endy, lenth3 = i
    vec_se = (endx - startx, endy - starty)
    # for i in range(len(storge_vh)):
    # 	storge_vh[i][0],storge_vh[i][1],storge_vh[i][2],storge_vh[i][3]=storge_vh[i][0]-minx+10,storge_vh[i][1]-miny+10,storge_vh[i][2]-minx+10,storge_vh[i][3]-miny+10
    return startx, starty, endx, endy, vec_se


def intersect(x_, y_, x0, y0, x1, y1, x2, y2, x3, y3, area):
    c_x_12, c_y_12 = findIntersection(x_, y_, x0, y0, x1, y1, x2, y2)
    c_x_01, c_y_01 = findIntersection(x3, y3, x2, y2, x0, y0, x1, y1)
    area_c_12 = polygon_area([[x0, y0], [x1, y1], [c_x_12, c_y_12]]) if c_x_12 else None
    area_c_01 = polygon_area([[x2, y2], [x1, y1], [c_x_01, c_y_01]]) if c_x_01 else None
    intersect_12 = c_x_12 and min(x1, x2) <= c_x_12 <= max(x1, x2) and min(y1, y2) <= c_y_12 <= max(y1, y2)
    intersect_01 = c_x_01 and min(x0, x1) <= c_x_01 <= max(x0, x1) and min(y0, y1) <= c_y_01 <= max(y0, y1)
    if area:
        intersect_12 = intersect_12 and (not c_x_01 or not (
                min(x0, x1) <= c_x_01 <= max(x0, x1) and min(y0, y1) <= c_y_01 <= max(y0,
                                                                                      y1)) or area_c_12 <= area_c_01)
        intersect_01 = intersect_01 and (not c_x_12 or not (
                min(x1, x2) <= c_x_12 <= max(x1, x2) and min(y1, y2) <= c_y_12 <= max(y1,
                                                                                      y2)) or area_c_01 < area_c_12)
    return c_x_12, c_y_12, c_x_01, c_y_01, area_c_12, area_c_01, intersect_12, intersect_01


def no_rect(approx, contour_area, box_area, vec_se, minx, miny, conti, pix):
    for i in range(len(approx) - 1, -1, -1):
        x1_, y1_ = approx[(i + 3) % len(approx)];
        x_, y_ = approx[(i + 2) % len(approx)];
        x0, y0 = approx[(i + 1) % len(approx)];
        x1, y1 = approx[i];
        x2, y2 = approx[i - 1];
        x3, y3 = approx[i - 2];
        x4, y4 = approx[i - 3]
        _, _, _, _, _, _, intersect_12, intersect_01 = intersect(x_, y_, x0, y0, x1, y1, x2, y2, x3, y3, False)

        A1_, B1_, C1_ = two_points_line(x1_, y1_, x_, y_, x1_, y1_);
        A_, B_, C_ = two_points_line(x_, y_, x0, y0, x_, y_);
        A0, B0, C0 = two_points_line(x0, y0, x1, y1, x0, y0);
        A1, B1, C1 = two_points_line(x1, y1, x2, y2, x1, y1);
        A2, B2, C2 = two_points_line(x2, y2, x3, y3, x2, y2);
        A3, B3, C3 = two_points_line(x3, y3, x4, y4, x3, y3)

        x_2, y_2 = Drop_point(A_, B_, C_, x2, y2);
        x30, y30 = Drop_point(A2, B2, C2, x0, y0);
        x1_2, y1_2 = Drop_point(A1_, B1_, C1_, x2, y2);
        x40, y40 = Drop_point(A3, B3, C3, x0, y0)
        try:
            angle_, angle0, angle, angle2, angle3 = cal_ang([x1_, y1_], [x_, y_], [x0, y0]), cal_ang([x_, y_], [x0, y0],
                                                                                                     [x1, y1]), cal_ang(
                [x0, y0], [x1, y1], [x2, y2]), cal_ang([x1, y1], [x2, y2], [x3, y3]), cal_ang([x2, y2], [x3, y3],
                                                                                              [x4, y4])
        except:
            continue
        angle_1__, angle__0, angle_01, angle_12, angle_23, angle_34 = cal_angle((x_ - x1_, y_ - y1_),
                                                                                vec_se), cal_angle((x0 - x_, y0 - y_),
                                                                                                   vec_se), cal_angle(
            (x1 - x0, y1 - y0), vec_se), cal_angle((x2 - x1, y2 - y1), vec_se), cal_angle((x3 - x2, y3 - y2),
                                                                                          vec_se), cal_angle(
            (x4 - x3, y4 - y3), vec_se)
        angle_2, angle03, angle2_, angle30 = cal_angle((x0 - x_, y0 - y_), (x2 - x1, y2 - y1), False), cal_angle(
            (x1 - x0, y1 - y0), (x3 - x2, y3 - y2), False), cal_angle((x2 - x_, y2 - y_), vec_se), cal_angle(
            (x3 - x0, y3 - y0), vec_se)
        angle1_2, angle04, angle21_, angle40 = cal_angle((x_ - x1_, y_ - y1_), (x2 - x1, y2 - y1), False), cal_angle(
            (x1 - x0, y1 - y0), (x4 - x3, y4 - y3), False), cal_angle((x2 - x1_, y2 - y1_), vec_se), cal_angle(
            (x4 - x0, y4 - y0), vec_se)
        min01, min12 = min(angle_01 - 0, 90 - angle_01), min(angle_12 - 0, 90 - angle_12)
        dis012, dis01__, dis201, dis234, dis134, dis_12 = abs(A1 * x0 + B1 * y0 + C1) / math.sqrt(
            A1 * A1 + B1 * B1), abs(A1_ * x0 + B1_ * y0 + C1_) / math.sqrt(A1_ * A1_ + B1_ * B1_), abs(
            A0 * x2 + B0 * y2 + C0) / math.sqrt(A0 * A0 + B0 * B0), abs(A3 * x2 + B3 * y2 + C3) / math.sqrt(
            A3 * A3 + B3 * B3), abs(A3 * x1 + B3 * y1 + C3) / math.sqrt(A3 * A3 + B3 * B3), abs(
            A1 * x_ + B1 * y_ + C1) / math.sqrt(A1 * A1 + B1 * B1)
        area1 = polygon_area([[x0, y0], [x1, y1], [x2, y2]])
        app1, app0, app2 = approx.copy(), approx.copy(), approx.copy();
        app1.pop(i), app0.pop((i + 1) % len(approx)), app2.pop(i - 1)
        if rayCasting([x1, y1], app1):
            if 70 <= angle <= 110 and min(min01, min12) <= 7 and area1 / contour_area > 0.02 and min(dis012,
                                                                                                     dis201) > 7 * pix and (
                    not intersect_12 and not intersect_01 or min01 <= 7 and min12 <= 7):
                conti = 0
                break
            elif angle <= 150 and angle_2 <= 7 and min12 <= 7 and polygon_area(
                    [[x0, y0], [x1, y1], [x2, y2], [x_2, y_2]]) / contour_area > 0.06 and dis012 > 7 * pix and min(
                angle2_ - 0, 90 - angle2_) > min(angle__0 - 0, 90 - angle__0, angle_12 - 0, 90 - angle_12) + 1:
                conti = 0
                break
            elif angle <= 150 and angle03 <= 7 and min01 <= 7 and polygon_area(
                    [[x0, y0], [x1, y1], [x2, y2], [x30, y30]]) / contour_area > 0.06 and dis201 > 7 * pix and min(
                angle30 - 0, 90 - angle30) > min(angle_01 - 0, 90 - angle_01, angle_23 - 0, 90 - angle_23) + 1:
                conti = 0
                break
            elif angle1_2 <= 7 and min12 <= 7 and polygon_area(
                    [[x_, y_], [x0, y0], [x1, y1], [x2, y2], [x1_2, y1_2]]) / contour_area > 0.08 and max(dis01__,
                                                                                                          dis012) < dis_12 and dis_12 > 7 * pix and min(
                angle21_ - 0, 90 - angle21_) > min(angle_1__ - 0, 90 - angle_1__, angle_12 - 0, 90 - angle_12) + 1:
                conti = 0
                break
            elif angle04 <= 7 and min01 <= 7 and polygon_area(
                    [[x0, y0], [x1, y1], [x2, y2], [x3, y3], [x40, y40]]) / contour_area > 0.08 and max(dis234,
                                                                                                        dis201) < dis134 and dis134 > 7 * pix and min(
                angle40 - 0, 90 - angle40) > min(angle_01 - 0, 90 - angle_01, angle_34 - 0, 90 - angle_34) + 1:
                conti = 0
                break
            elif angle <= 150 and min01 <= 7 and abs(90 - angle0) <= 15 and polygon_area(
                    [[x0, y0], [x1, y1], [x2, y2], [x_2, y_2]]) / contour_area > 0.07 and dis201 > 7 * pix:
                conti = 0
                break
            elif angle <= 150 and min12 <= 7 and abs(90 - angle2) <= 15 and polygon_area(
                    [[x0, y0], [x1, y1], [x2, y2], [x30, y30]]) / contour_area > 0.07 and dis012 > 7 * pix:
                conti = 0
                break
            elif rayCasting([x0, y0], app0) and rayCasting([x2, y2], app2) and polygon_area(
                    [[x_, y_], [x0, y0], [x1, y1], [x2, y2], [x3, y3]]) / contour_area > 0.04 and (
                    abs(90 - angle_) <= 25 or abs(90 - angle3) <= 20):
                conti = 0
                break
            elif rayCasting([x2, y2], app2) and polygon_area(
                    [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]) / contour_area > 0.04 and (
                    abs(90 - angle0) <= 25 or abs(90 - angle3) <= 20):
                conti = 0
                break
    return conti


def zhu_rect(startx, starty, endx, endy, vec_se, approx, box0, box0_area):
    A, B, C = two_points_line(startx, starty, endx, endy, approx[0][0], approx[0][1])
    pd, pl, pr = [], [], []

    for i in range(len(approx) - 1, -1, -1):
        x, y = approx[i]
        xd, yd = Drop_point(A, B, C, x, y)
        pd.append([xd, yd])
    if cal_angle(vec_se, (1, 0)) <= 45:
        pd.sort(key=lambda x: x[0], reverse=False)
    else:
        pd.sort(key=lambda x: x[1], reverse=False)
    x1, y1 = pd[0]
    x2, y2 = pd[-1]
    Al, Bl, Cl = two_points_vline(startx, starty, endx, endy, x1, y1)
    Ar, Br, Cr = two_points_vline(startx, starty, endx, endy, x2, y2)

    for i in range(len(approx) - 1, -1, -1):
        x, y = approx[i]
        xl, yl = Drop_point(Al, Bl, Cl, x, y)
        xr, yr = Drop_point(Ar, Br, Cr, x, y)
        pl.append([xl, yl])
        pr.append([xr, yr])

    if cal_angle(vec_se, (1, 0)) <= 45:
        pl.sort(key=lambda x: x[1], reverse=False)
        pr.sort(key=lambda x: x[1], reverse=False)
    else:
        pl.sort(key=lambda x: x[0], reverse=False)
        pr.sort(key=lambda x: x[0], reverse=False)

    box0 = np.int0(np.array([pl[0], pr[0], pr[-1], pl[-1]]))
    box0_area = math.sqrt((pl[0][1] - pl[-1][1]) ** 2 + (pl[0][0] - pl[-1][0]) ** 2) * math.sqrt(
        (pr[0][1] - pl[0][1]) ** 2 + (pr[0][0] - pl[0][0]) ** 2)
    return box0, box0_area


def regular(img, seg_type, pix):
    pix, pix_2 = 0.5 / pix, 0.25 / pix / pix
    im = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    # 去除最小面积及最小空洞区域
    img = img > 0
    if seg_type in ['bfp', "resident"]:
        img = morphology.remove_small_objects(img, min_size=25, connectivity=1, in_place=True)
        img = morphology.remove_small_holes(img, area_threshold=60, connectivity=1,
                                            in_place=True) if seg_type == 'bfp' else morphology.remove_small_holes(img,
                                                                                                                   area_threshold=900,
                                                                                                                   connectivity=1,
                                                                                                                   in_place=True)
    elif seg_type in ['road']:
        img = morphology.remove_small_objects(img, min_size=60, connectivity=1, in_place=True)
        img = morphology.remove_small_holes(img, area_threshold=60, connectivity=1, in_place=True)
    img = ((img + 0) * 255).astype(np.uint8)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # RETR_TREE  CHAIN_APPROX_NONE
    final, inter_outer = [], []
    ostartx = ostarty = oendx = oendy = ovec_se = index = False
    #######################################################################################遍历轮廓
    for c in range(len(contours)):
        show, step_show, conti = False, False, False
        approx = cv2.approxPolyDP(contours[c], 1.5, True) if seg_type in ['bfp', 'resident',
                                                                          'road'] else cv2.approxPolyDP(contours[c], 1,
                                                                                                        True)
        if seg_type not in ['bfp', 'resident']:
            approx = approx.reshape(-1, 2)
            final.append(approx)
            inter_outer.append([c, hierarchy[0, c, 3]])
            continue

        if approx.shape[0] < 3: continue
        minx, maxx, miny, maxy = min(approx[:, 0, 0]), max(approx[:, 0, 0]), min(approx[:, 0, 1]), max(approx[:, 0, 1])
        approx = approx.reshape(-1, 2).tolist()[::-1]
        contour = contours[c].reshape(-1, 2).tolist()
        appro = approx.copy()
        contour_area = cv2.contourArea(contours[c])
        rect = cv2.minAreaRect(contours[c])
        box = np.int0(cv2.boxPoints(rect))
        xlu, ylu = box[0, :];
        xru, yru = box[1, :];
        xrd, yrd = box[2, :]
        box_area = math.sqrt((yru - ylu) ** 2 + (xru - xlu) ** 2) * math.sqrt((yrd - yru) ** 2 + (xrd - xru) ** 2)
        if hierarchy[0, c, 3] == -1:
            startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True, seg_type)
            box0, box0_area = zhu_rect(startx, starty, endx, endy, vec_se, contour, box, box_area)
        elif ostartx and c == index + 1:
            index = c
            startx, starty, endx, endy, vec_se = ostartx, ostarty, oendx, oendy, ovec_se
            box0, box0_area = zhu_rect(startx, starty, endx, endy, vec_se, contour, box, box_area)
        else:
            continue

        if hierarchy[0, c, 3] == -1:
            if contour_area / box_area > 0.9:
                conti = 1 if (box_area - contour_area) < (300 * pix_2) else 0
            elif contour_area / box_area > 0.8:
                conti = no_rect(approx, contour_area, box_area, vec_se, minx, miny, 1, pix) if (
                                                                                                       box_area - contour_area) < (
                                                                                                       500 * pix_2) else 0
                if conti == 1:
                    conti = 2 if box_area / box0_area >= 0.93 else 1
            elif contour_area / box_area > 0.7 and len(approx) == 6:
                conti = no_rect(approx, contour_area, box_area, vec_se, minx, miny, 1, pix) if (
                                                                                                       box_area - contour_area) < (
                                                                                                       500 * pix_2) else 0
                if conti == 1:
                    conti = 2 if box_area / box0_area >= 0.93 else 1
            elif contour_area / box_area > 0.6 and len(approx) == 5 or contour_area / box_area > 0.55 and len(
                    approx) <= 4:
                if (box_area - contour_area) < (300 * pix_2):
                    conti = 2 if box_area / box0_area >= 0.93 else 1
                else:
                    conti = 0
        else:
            if contour_area / box0_area > 0.9:
                conti = 2
            elif contour_area / box0_area > 0.8:
                conti = no_rect(approx, contour_area, box0_area, vec_se, minx, miny, 2, pix) if len(approx) >= 6 else 2
            elif contour_area / box0_area > 0.75 and len(approx) == 6:
                conti = no_rect(approx, contour_area, box0_area, vec_se, minx, miny, 2, pix)
            elif contour_area / box0_area > 0.75 and len(approx) <= 5:
                conti = 2

        if conti == 1:
            final.append(box)
            inter_outer.append([c, hierarchy[0, c, 3]])
            continue
        elif conti == 2:
            final.append(box0)
            inter_outer.append([c, hierarchy[0, c, 3]])
            continue

        #########################几条线本为一条线
        jump = 0
        l1 = cv2.arcLength(np.round(np.array(approx)).astype(np.int32), True) / len(approx)
        for i in range(len(approx) - 1, -1, -1):
            if len(approx) < 4:
                break
            if jump != 0:
                jump -= 1
                continue
            x1_, y1_ = approx[(i + 3) % len(approx)];
            x_, y_ = approx[(i + 2) % len(approx)];
            x0, y0 = approx[(i + 1) % len(approx)];
            x1, y1 = approx[i];
            x2, y2 = approx[i - 1];
            x3, y3 = approx[i - 2];
            x4, y4 = approx[i - 3]
            vec__0, vec_01, vec_12, vec_23, vec__1, vec_02, vec_13, vec__3, vec_03, vec__2, vec_b = (
                                                                                                        x0 - x_,
                                                                                                        y0 - y_), (
                                                                                                        x1 - x0,
                                                                                                        y1 - y0), (
                                                                                                        x2 - x1,
                                                                                                        y2 - y1), (
                                                                                                        x3 - x2,
                                                                                                        y3 - y2), (
                                                                                                        x1 - x_,
                                                                                                        y1 - y_), (
                                                                                                        x2 - x0,
                                                                                                        y2 - y0), (
                                                                                                        x3 - x1,
                                                                                                        y3 - y1), (
                                                                                                        x3 - x_,
                                                                                                        y3 - y_), (
                                                                                                        x3 - x0,
                                                                                                        y3 - y0), (
                                                                                                        x2 - x_,
                                                                                                        y2 - y_), (
                                                                                                        xru - xlu,
                                                                                                        yru - ylu)
            try:
                angle0, angle1, angle2, angle_1, angle_013, angle__12 = cal_ang([x_, y_], [x0, y0], [x1, y1]), cal_ang(
                    [x0, y0], [x1, y1], [x2, y2]), cal_ang([x1, y1], [x2, y2], [x3, y3]), cal_ang([x_, y_], [x1, y1],
                                                                                                  [x3, y3]), cal_ang(
                    [x0, y0], [x1, y1], [x3, y3]), cal_ang([x_, y_], [x1, y1], [x2, y2])
            except:
                continue
            dis__0, dis_01, dis_12, dis_23 = math.sqrt((y0 - y_) ** 2 + (x_ - x0) ** 2), math.sqrt(
                (y1 - y0) ** 2 + (x1 - x0) ** 2), math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2), math.sqrt(
                (y3 - y2) ** 2 + (x3 - x2) ** 2)
            angle__0, angle_01, angle_12, angle_23, angle__1, angle_02, angle_13, angle__3, angle_03, angle__2, angleb_1, angleb13, angleb_3 = cal_angle(
                vec__0, vec_se), cal_angle(vec_01, vec_se), cal_angle(vec_12, vec_se), cal_angle(vec_23,
                                                                                                 vec_se), cal_angle(
                vec__1, vec_se), cal_angle(vec_02, vec_se), cal_angle(vec_13, vec_se), cal_angle(vec__3,
                                                                                                 vec_se), cal_angle(
                vec_03, vec_se), cal_angle(vec__2, vec_se), cal_angle(vec__1, vec_b), cal_angle(vec_13,
                                                                                                vec_b), cal_angle(
                vec__3, vec_b)
            min_1, min02, min13, min_3, min03 = min(90 - angle__0, angle__0 - 0, 90 - angle_01, angle_01 - 0), min(
                90 - angle_01, angle_01 - 0, 90 - angle_12, angle_12 - 0), min(90 - angle_12, angle_12 - 0,
                                                                               90 - angle_23, angle_23 - 0), min(
                90 - angle__0, angle__0 - 0, 90 - angle_01, angle_01 - 0, 90 - angle_12, angle_12 - 0, 90 - angle_23,
                angle_23 - 0), min(angle_01 - 0, 90 - angle_01, angle_12 - 0, 90 - angle_12, angle_23 - 0,
                                   90 - angle_23)
            min1_, min31, min20 = min(angle__0 - 0, 90 - angle__0) * dis__0 / (dis__0 + dis_01) + min(angle_01 - 0,
                                                                                                      90 - angle_01) * dis_01 / (
                                          dis__0 + dis_01), min(angle_12 - 0, 90 - angle_12) * dis_12 / (
                                          dis_12 + dis_23) + min(angle_23 - 0, 90 - angle_23) * dis_23 / (
                                          dis_12 + dis_23), min(angle_01 - 0, 90 - angle_01) * dis_01 / (
                                          dis_01 + dis_12) + min(angle_12 - 0, 90 - angle_12) * dis_12 / (
                                          dis_01 + dis_12)
            angle__x, angle_0x, angle_1x, angle_2x = cal_angle(vec__0, (1, 0)), cal_angle(vec_01, (1, 0)), cal_angle(
                vec_12, (1, 0)), cal_angle(vec_23, (1, 0))
            min__1, min_02, min_13, min_x, min0x, min1x, min2x = min(angle__1, 90 - angle__1), min(angle_02,
                                                                                                   90 - angle_02), min(
                angle_13, 90 - angle_13), min(angle__x, 90 - angle__x), min(angle_0x, 90 - angle_0x), min(angle_1x,
                                                                                                          90 - angle_1x), min(
                angle_2x, 90 - angle_2x)
            area_0, area_1, area_2 = polygon_area([[x_, y_], [x0, y0], [x1, y1]]), polygon_area(
                [[x0, y0], [x1, y1], [x2, y2]]), polygon_area([[x1, y1], [x2, y2], [x3, y3]])
            app_, app0, app1, app2, app3 = approx.copy(), approx.copy(), approx.copy(), approx.copy(), approx.copy();
            app_.pop((i + 2) % len(approx));
            app0.pop((i + 1) % len(approx));
            app1.pop(i);
            app2.pop(i - 1);
            app3.pop(i - 2)
            A0, B0, C0 = two_points_line(x0, y0, x1, y1, x0, y0);
            A1, B1, C1 = two_points_line(x1, y1, x2, y2, x1, y1)
            vec_1__, vec_34 = (x_ - x1_, y_ - y1_), (x4 - x3, y4 - y3)
            d0, d2 = abs(A1 * x0 + B1 * y0 + C1) / math.sqrt(A1 * A1 + B1 * B1), abs(
                A0 * x2 + B0 * y2 + C0) / math.sqrt(A0 * A0 + B0 * B0)
            angle_0123, angle__012 = cal_angle(vec_01, vec_23, False), cal_angle(vec__0, vec_12, False)
            min01, min12, min_03, min__2 = min(angle_01 - 0, 90 - angle_01), min(angle_12 - 0, 90 - angle_12), min(
                angle_03 - 0, 90 - angle_03), min(angle__2 - 0, 90 - angle__2)
            if 145 <= angle0 and 145 <= angle1 and 145 <= angle2 and rayCasting([x0, y0], app0) != rayCasting([x1, y1],
                                                                                                              app1) != rayCasting(
                [x2, y2], app2):
                conti = False
                if not rayCasting([x1, y1], app1):
                    if min(90 - angle__3, angle__3 - 0) <= min(min_3, min(90 - angle__1, angle__1 - 0, 90 - angle_13,
                                                                          angle_13 - 0)) + 2:
                        conti = -1
                    elif angle_1 >= 175 and min(90 - angle__3, angle__3 - 0) <= min_3 + 2:
                        conti = -1
                    if conti == -1:
                        if i + 1 == len(approx): jump += 1
                        jump += 1
                        approx.remove([x0, y0])
                        approx.remove([x1, y1])
                        approx.remove([x2, y2])
                        if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                              seg_type)
                        continue
                    if min(90 - angle__1, angle__1 - 0) <= min_1 + 2:
                        conti = 1
                    elif angle_1 >= 175 and (min_1 > 5 or min(angle__1 - 0, 90 - angle__1, angle__3 - 0,
                                                              90 - angle__3) <= min_1 + 3 if angle0 < 157 else min1_):
                        conti = 1
                    elif angle0 >= 170 or angle0 >= 160 and (min_1 > 5 or min(angle__1 - 0, 90 - angle__1) <= min1_):
                        conti = 1
                    if conti == 1:
                        if i + 1 == len(approx): jump += 1
                        approx.remove([x0, y0])
                        if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                              seg_type)

                    if min(90 - angle_13, angle_13 - 0) <= min13 + 2:
                        conti = 2
                    elif angle_1 >= 175 and (min13 > 5 or min(angle_13 - 0, 90 - angle_13, angle__3 - 0,
                                                              90 - angle__3) <= min13 + 3 if angle2 < 157 else min31):
                        conti = 2
                    elif angle2 >= 170 or angle2 >= 160 and (min13 > 5 or min(angle_13 - 0, 90 - angle_13) <= min31):
                        conti = 2
                    if conti == 2:
                        jump += 1
                        approx.remove([x2, y2])
                        if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                              seg_type)
                    if step_show: show_result(appro, approx, x1, y1, True)

                else:
                    if min(angle__3 - 0, 90 - angle__3) <= min(min_3, min(angle_02 - 0, 90 - angle_02)) + 2:
                        cx_01, cy_01 = x0 + (x_ - x0) / 4, y0 + (y_ - y0) / 4
                        cx_12, cy_12 = x2 + (x3 - x2) / 4, y2 + (y3 - y2) / 4
                        Ac, Bc, Cc = two_points_line(cx_01, cy_01, cx_12, cy_12, cx_01, cy_01)
                        x__, y__ = Drop_point(Ac, Bc, Cc, x_, y_)
                        x_3, y_3 = Drop_point(Ac, Bc, Cc, x3, y3)
                        approx[(i + 2) % len(approx)] = [x__, y__]
                        approx[i - 2] = [x_3, y_3]
                        if i + 1 == len(approx): jump += 1
                        jump += 1
                        approx.remove([x0, y0])
                        approx.remove([x1, y1])
                        approx.remove([x2, y2])
                        if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                              seg_type)
                        if step_show: show_result(appro, approx, x1, y1, True)
                    elif min(angle_02 - 0, 90 - angle_02) <= min02 + 2 or angle1 >= 170 or angle1 >= 160 and (
                            min02 > 5 or min(angle_02 - 0, 90 - angle_02) <= min20):
                        approx.pop(i)
                        if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                              seg_type)
                        if step_show: show_result(appro, approx, x1, y1, True)


            elif angle1 >= 160 and angle2 >= 160 and rayCasting([x1, y1], app1) != rayCasting([x2, y2], app2) and max(
                    dis_01, dis_12, dis_23) / min(dis_01, dis_12, dis_23) <= 2.5:
                if cal_angle(vec_01, vec_23) <= 3 and min(cal_angle(vec_01, (1, 0)) - 0, 90 - cal_angle(vec_01, (1, 0)),
                                                          cal_angle(vec_23, (1, 0)) - 0,
                                                          90 - cal_angle(vec_23, (1, 0))) <= 3:
                    continue
                if min(angle_03 - 0, 90 - angle_03) <= min03 + 2:
                    conti = 3
                elif rayCasting([x1, y1], app1):
                    if cal_ang([x0, y0], [x2, y2], [x3, y3]) >= 175 and (
                            min02 > 5 or min(angle_02 - 0, 90 - angle_02, angle_03 - 0, 90 - angle_03) <= min20):
                        conti = 3
                    elif angle1 >= 170 or min02 > 5 or min(angle_02 - 0, 90 - angle_02) <= min20:
                        approx.pop(i)
                        if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                              seg_type)
                        if step_show: show_result(appro, approx, x1, y1, True)
                elif rayCasting([x2, y2], app2):
                    if cal_ang([x0, y0], [x1, y1], [x3, y3]) >= 175 and (
                            min13 > 5 or min(angle_13 - 0, 90 - angle_13, angle_03 - 0, 90 - angle_03) <= min31):
                        conti = 3
                    elif angle2 >= 170 or min13 > 5 or min(angle_13 - 0, 90 - angle_13) <= min31:
                        jump += 1
                        approx.pop(i - 1)
                        if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                              seg_type)
                        if step_show: show_result(appro, approx, x1, y1, True)

                if conti == 3:
                    if rayCasting([x1, y1], app1):
                        cx_01, cy_01 = (x0 + x2) / 2, (y0 + y2) / 2
                        cx_12, cy_12 = (x2 + x3) / 2, (y2 + y3) / 2
                    else:
                        cx_01, cy_01 = (x0 + x1) / 2, (y0 + y1) / 2
                        cx_12, cy_12 = (x1 + x3) / 2, (y1 + y3) / 2
                    Ac, Bc, Cc = two_points_line(cx_01, cy_01, cx_12, cy_12, cx_01, cy_01)
                    x_0, y_0 = Drop_point(Ac, Bc, Cc, x0, y0)
                    x_3, y_3 = Drop_point(Ac, Bc, Cc, x3, y3)
                    approx[(i + 1) % len(approx)] = [x_0, y_0]
                    approx[i - 2] = [x_3, y_3]
                    approx.pop(i)
                    approx.pop(i - 1)
                    jump += 1
                    if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                          seg_type)
                    if step_show: show_result(appro, approx, x1, y1, True)
            elif rayCasting([x1, y1], app1) and d2 <= 5 and angle_0123 <= 45 and angle_013 > max(173,
                                                                                                 180 - angle_0123) and min_03 <= min(
                7, min_02) and dis_01 > 0.8 * math.sqrt((y3 - y1) ** 2 + (x3 - x1) ** 2):
                if area_judg(polygon_area([[x1, y1], [x2, y2], [x3, y3]]), contour_area, -1, seg_type):
                    approx.pop(i - 1)
                    jump += 1
                else:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    Ac, Bc, Cc = two_points_line(x0, y0, x3, y3, cx, cy) if min_03 <= min01 else two_points_line(x0, y0,
                                                                                                                 x1, y1,
                                                                                                                 cx, cy)
                    x_0, y_0 = Drop_point(Ac, Bc, Cc, x0, y0)
                    x_3, y_3 = Drop_point(Ac, Bc, Cc, x3, y3)
                    area_c = union([[x1, y1], [x2, y2], [x3, y3]], [[x0, y0], [x_0, y_0], [x_3, y_3], [x3, y3]])
                    angle_new, angle_old = cal_angle((x_3 - x_0, y_3 - y_0), (x4 - x_3, y4 - y_3), False), cal_angle(
                        vec_01, vec_34, False)
                    if area_judg(area_c, contour_area, 1, seg_type) and (angle_old > 7 or angle_new <= angle_old):
                        approx[(i + 1) % len(approx)] = [x_0, y_0]
                        approx[i - 2] = [x_3, y_3]
                        approx.pop(i)
                        approx.pop(i - 1)
                        jump += 1
                        if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                              seg_type)
            elif rayCasting([x1, y1], app1) and d0 <= 5 and angle__012 <= 45 and angle__12 > max(173,
                                                                                                 180 - angle__012) and min__2 <= min(
                7, min_02) and dis_12 > 0.8 * math.sqrt((y1 - y_) ** 2 + (x1 - x_) ** 2):
                if area_judg(polygon_area([[x_, y_], [x0, y0], [x1, y1]]), contour_area, -1, seg_type):
                    if i + 1 == len(approx): jump += 1
                    approx.pop((i + 1) % len(approx))
                else:
                    cx, cy = (x1 + x0) / 2, (y1 + y0) / 2
                    Ac, Bc, Cc = two_points_line(x_, y_, x2, y2, cx, cy) if min__2 <= min12 else two_points_line(x1, y1,
                                                                                                                 x2, y2,
                                                                                                                 cx, cy)
                    x__, y__ = Drop_point(Ac, Bc, Cc, x_, y_)
                    x_2, y_2 = Drop_point(Ac, Bc, Cc, x2, y2)
                    area_c = union([[x_, y_], [x0, y0], [x1, y1]], [[x_, y_], [x__, y__], [x_2, y_2], [x2, y2]])
                    angle_old, angle_new = cal_angle(vec_1__, vec_12, False), cal_angle((x__ - x1_, y__ - y1_),
                                                                                        (x_2 - x__, y_2 - y__), False)
                    if area_judg(area_c, contour_area, 1, seg_type) and (angle_old > 7 or angle_new <= angle_old):
                        approx[(i + 2) % len(approx)] = [x__, y__]
                        approx[i - 1] = [x_2, y_2]
                        if i + 1 == len(approx): jump += 1
                        approx.remove([x0, y0])
                        approx.remove([x1, y1])
                        if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                              seg_type)
            elif rayCasting([x0, y0], app0) and rayCasting([x1, y1], app1) and rayCasting([x2, y2],
                                                                                          app2) and not rayCasting(
                [x_, y_], app_) and not rayCasting([x3, y3], app3):
                if min__1 <= min(min_1, 7) and min(min_x, min0x) > 4 and area_judg(area_0, contour_area, 2,
                                                                                   seg_type) and (
                        abs(angle0 - 90) > 20 or min(dis__0, dis_01) <= 5):
                    if i + 1 == len(approx): jump += 1
                    approx.pop((i + 1) % len(approx))
                    if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                          seg_type)
                    if step_show: show_result(appro, approx, x1, y1, True)
                elif min_13 <= min(min13, 7) and min(min1x, min2x) > 4 and area_judg(area_2, contour_area, 2,
                                                                                     seg_type) and (
                        abs(angle2 - 90) > 20 or min(dis_23, dis_12) <= 5):
                    approx.pop(i - 1)
                    jump += 1
                    if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                          seg_type)
                    if step_show: show_result(appro, approx, x1, y1, True)
                elif min_02 <= min(min02, 7) and min(min0x, min1x) > 4 and area_judg(area_1, contour_area, 2,
                                                                                     seg_type) and (
                        abs(angle1 - 90) > 20 or min(dis_01, dis_12) <= 5):
                    approx.pop(i)
                    if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                          seg_type)
                    if step_show: show_result(appro, approx, x1, y1, True)
            elif rayCasting([x1, y1], app1) and rayCasting([x2, y2], app2) and not rayCasting([x0, y0],
                                                                                              app0) and not rayCasting(
                [x3, y3], app3):
                if min_02 < min(min02, 7) and min(min0x, min1x) > 4 and area_judg(area_1, contour_area, 2,
                                                                                  seg_type) and abs(angle1 - 90) > 20:
                    approx.pop(i)
                    if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                          seg_type)
                    if step_show: show_result(appro, approx, x1, y1, True)
                elif min_13 < min(min13, 7) and min(min1x, min2x) > 4 and area_judg(area_2, contour_area, 2,
                                                                                    seg_type) and abs(angle2 - 90) > 20:
                    approx.pop(i - 1)
                    jump += 1
                    if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                          seg_type)
                    if step_show: show_result(appro, approx, x1, y1, True)
            elif rayCasting([x1, y1], app1) and not (not rayCasting([x0, y0], app0) and not rayCasting([x2, y2], app2)):
                if min_02 <= min(min02, 7) and min(min0x, min1x) > 4 and area_judg(area_1, contour_area, 2,
                                                                                   seg_type) and (
                        abs(angle1 - 90) > 20 or min(dis_01, dis_12) <= 5):
                    approx.pop(i)
        ##############################################################################################################################旋转
        approx, show = spin(approx, contour, contour_area, startx, starty, endx, endy, vec_se, seg_type, True)
        ###############################################################################################################################遍历多边形
        l1 = cv2.arcLength(np.round(np.array(approx)).astype(np.int32), True) / len(approx)
        jump, skip = 0, 0
        for i in range(len(approx) - 1, -1, -1):
            if jump != 0:
                jump -= 1
                continue
            if len(approx) < 4: break
            for f in range(i + 3, i - 4, -1):
                if all(np.round(approx[f % len(approx)]) == np.round(approx[(f - 1) % len(approx)])):
                    if 0 <= f % len(approx) <= i:
                        approx.pop(f % len(approx))
                        skip = True
                        break
                    else:
                        approx.pop(f % len(approx))
            if skip or len(approx) < 4:
                skip = 0
                continue
            ##################################################################################定义变量
            x1_, y1_ = approx[(i + 3) % len(approx)];
            x_, y_ = approx[(i + 2) % len(approx)];
            x0, y0 = approx[(i + 1) % len(approx)];
            x1, y1 = approx[i];
            x2, y2 = approx[i - 1];
            x3, y3 = approx[i - 2];
            x4, y4 = approx[i - 3]
            dis_1__, dis__0, dis_01, dis_12, dis_02, dis_23, dis_34 = math.sqrt(
                (y1_ - y_) ** 2 + (x1_ - x_) ** 2), math.sqrt((y0 - y_) ** 2 + (x_ - x0) ** 2), math.sqrt(
                (y1 - y0) ** 2 + (x1 - x0) ** 2), math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2), math.sqrt(
                (y2 - y0) ** 2 + (x2 - x0) ** 2), math.sqrt((y3 - y2) ** 2 + (x3 - x2) ** 2), math.sqrt(
                (y4 - y3) ** 2 + (x4 - x3) ** 2)  # 线长
            proportion_01, proportion_12 = dis_01 / (dis_01 + dis_12), dis_12 / (dis_01 + dis_12)
            A1_, B1_, C1_ = two_points_line(x1_, y1_, x_, y_, x_, y_);
            A_, B_, C_ = two_points_line(x_, y_, x0, y0, x0, y0);
            A0, B0, C0 = two_points_line(x0, y0, x1, y1, x0, y0);
            A1, B1, C1 = two_points_line(x1, y1, x2, y2, x1, y1);
            A2, B2, C2 = two_points_line(x2, y2, x3, y3, x2, y2);
            A3, B3, C3 = two_points_line(x3, y3, x4, y4, x3, y3)
            d1_23, d0_12 = abs(A2 * x1 + B2 * y1 + C2) / math.sqrt(A2 * A2 + B2 * B2), abs(
                A1 * x0 + B1 * y0 + C1) / math.sqrt(A1 * A1 + B1 * B1)
            try:
                angle__, angle_0, angle, angle_2, angle_3, angle__02, angle_023, angle_013 = cal_ang([x1_, y1_],
                                                                                                     [x_, y_],
                                                                                                     [x0, y0]), cal_ang(
                    [x_, y_], [x0, y0], [x1, y1]), cal_ang([x0, y0], [x1, y1], [x2, y2]), cal_ang([x1, y1], [x2, y2],
                                                                                                  [x3, y3]), cal_ang(
                    [x2, y2], [x3, y3], [x4, y4]), cal_ang((x_, y_), (x0, y0), (x2, y2)), cal_ang((x0, y0), (x2, y2),
                                                                                                  (x3, y3)), cal_ang(
                    (x0, y0), (x1, y1), (x3, y3))  # 三点夹角
            except:
                continue
            vec_1__, vec__0, vec_01, vec_12, vec_23, vec_34, vec_02 = (x_ - x1_, y_ - y1_), (x0 - x_, y0 - y_), (
                x1 - x0, y1 - y0), (x2 - x1, y2 - y1), (x3 - x2, y3 - y2), (x4 - x3, y4 - y3), (
                                                                      x2 - x0, y2 - y0)  # 01,12向量
            angle_1__, angle__0, angle_01, angle_12, angle_23, angle_34, angle_02 = cal_angle(vec_1__,
                                                                                              vec_se), cal_angle(vec__0,
                                                                                                                 vec_se), cal_angle(
                vec_01, vec_se), cal_angle(vec_12, vec_se), cal_angle(vec_23, vec_se), cal_angle(vec_34,
                                                                                                 vec_se), cal_angle(
                vec_02, vec_se)  # 01,02与主方向夹角
            angle_1_4, angle__4, angle_14, angle__3, angle_03, angle__2, angle_30, angle_2_ = cal_angle(vec_1__, vec_34,
                                                                                                        half=False), cal_angle(
                vec__0, vec_34), cal_angle(vec_12, vec_34), cal_angle(vec__0, vec_23, half=False), cal_angle(vec_01,
                                                                                                             vec_23), cal_angle(
                vec__0, vec_12), cal_angle((x3 - x0, y3 - y0), vec_se), cal_angle((x2 - x_, y2 - y_), vec_se)
            angle__x, angle_0x, angle_1x, angle_2x, angle_02x, angle_sex = cal_angle(vec__0, (1, 0)), cal_angle(vec_01,
                                                                                                                (1,
                                                                                                                 0)), cal_angle(
                vec_12, (1, 0)), cal_angle(vec_23, (1, 0)), cal_angle(vec_02, (1, 0)), cal_angle(vec_se, (1, 0))
            h0, v0 = horv(approx, (i + 1) % len(approx), vec_01, dis_01, l1, False);
            h1, v1 = horv(approx, i, vec_12, dis_12, l1, False);
            h2, v2 = horv(approx, i - 1, vec_23, dis_23, l1, False)
            area_1 = polygon_area([[x0, y0], [x1, y1], [x2, y2]])  # 三点面积
            min1__, min34 = min(angle_1__, 90 - angle_1__), min(angle_34, 90 - angle_34)
            min_0, min01, min12, min23, min02, min2_, min30 = min(angle__0 - 0, 90 - angle__0), min(angle_01 - 0,
                                                                                                    90 - angle_01), min(
                angle_12 - 0, 90 - angle_12), min(angle_23 - 0, 90 - angle_23), min(angle_02 - 0, 90 - angle_02), min(
                angle_2_ - 0, 90 - angle_2_), min(angle_30 - 0, 90 - angle_30)
            min20, min0x, min1x, min02x, minsex = min(90 - angle_01, angle_01 - 0, 90 - angle_12, angle_12 - 0), min(
                angle_0x - 0, 90 - angle_0x), min(angle_1x - 0, 90 - angle_1x), min(angle_02x - 0, 90 - angle_02x), min(
                angle_sex - 0, 90 - angle_sex)
            app__, app_0, appr, app_2, app_3, app_4 = approx.copy(), approx.copy(), approx.copy(), approx.copy(), approx.copy(), approx.copy()
            app__.pop((i + 2) % len(approx));
            app_0.pop((i + 1) % len(approx));
            appr.pop(i);
            app_2.pop(i - 1);
            app_3.pop(i - 2);
            app_4.pop(i - 3)
            ########################################################################################################################################################填充内角
            if rayCasting([x1, y1], appr):
                if angle > 165:
                    approx.pop(i)
                    if step_show: show_result(appro, approx, x1, y1, True)
                    if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                          seg_type)
                    continue
                elif not rayCasting([x2, y2], app_2) and not rayCasting([x0, y0], app_0):
                    conti = None
                    c_x_12, c_y_12, c_x_01, c_y_01, area_c_12, area_c_01, intersect_12, intersect_01 = intersect(x_, y_,
                                                                                                                 x0, y0,
                                                                                                                 x1, y1,
                                                                                                                 x2, y2,
                                                                                                                 x3, y3,
                                                                                                                 True)
                    diff_01, diff_12 = min(abs(90 - angle__02), abs(180 - angle__02)) <= min(abs(90 - angle_0),
                                                                                             abs(180 - angle_0),
                                                                                             15), min(
                        abs(90 - angle_023), abs(180 - angle_023)) <= min(abs(90 - angle_2), abs(180 - angle_2), 15)
                    diff_015, diff_125 = min(abs(90 - angle__02), abs(180 - angle__02)) <= min(abs(90 - angle_0),
                                                                                               abs(180 - angle_0),
                                                                                               7) and dis__0 > 7, min(
                        abs(90 - angle_023), abs(180 - angle_023)) <= min(abs(90 - angle_2), abs(180 - angle_2),
                                                                          7) and dis_23 > 7
                    x_2, y_2 = Drop_point(A0, B0, C0, x2, y2)
                    x_0, y_0 = Drop_point(A1, B1, C1, x0, y0)
                    area_2, area_0 = polygon_area([[x_2, y_2], [x2, y2], [x1, y1]]), polygon_area(
                        [[x_0, y_0], [x0, y0], [x1, y1]])
                    p201, p012 = min01 <= min(min12, 15) and lineup(x0, y0, x1, y1, x_2, y_2), min12 <= min(min01,
                                                                                                            15) and lineup(
                        x1, y1, x2, y2, x_0, y_0)
                    i201 = intersect_01 and min23 <= min12 and (
                            (c_x_01, c_y_01) == (x2, y2) or (c_x_01, c_y_01) == (x0, y0) or cal_ang((x0, y0), (
                        c_x_01, c_y_01), (x2, y2)) <= 115 or min23 <= min01) and (
                                   (c_x_01, c_y_01) != (x0, y0) or min23 <= min01)
                    i012 = intersect_12 and min_0 <= min01 and (
                            (c_x_12, c_y_12) == (x0, y0) or (c_x_12, c_y_12) == (x2, y2) or cal_ang((x0, y0), (
                        c_x_12, c_y_12), (x2, y2)) <= 115 or min_0 <= min12) and (
                                   (c_x_12, c_y_12) != (x2, y2) or min_0 <= min12)
                    beter02 = diff_01 and diff_12 and min20 > 7 and (dis__0 + dis_23) >= 0.7 * (
                            dis_01 + dis_12) or min02 <= min20 or min02 == min20 + 1 and max(dis_01, dis_12) / min(
                        dis_01, dis_12) < 3
                    skip = (angle_03 <= 5 and min(dis_01, dis_23) >= 15 and d1_23 > 6 or angle__2 <= 5 and min(dis__0,
                                                                                                               dis_12) >= 15 and d0_12 > 6) and (
                                   min20 <= 5 or min20 <= 7 and min02x > min(min0x, min1x, 5))
                    if angle > 45 and (
                            angle_03 < 15 and min(dis_01, dis_23) >= 8 and d1_23 > 6 or angle__2 < 15 and min(dis__0,
                                                                                                              dis_12) >= 8 and d0_12 > 6) and min02 > min20 + 1:
                        conti = False
                    elif 70 < angle < 110 and 70 < angle_0 < 110 and 70 < angle_2 < 110 and min02 > min20:
                        conti = 5 if p201 else 4 if p012 else False
                    elif angle > 45 and (85 <= angle_0 <= 95 and min01 < 10 and not i201 and min(abs(90 - angle__02),
                                                                                                 180 - angle__02) > 10 or 85 <= angle_2 <= 95 and min12 < 10 and not i012 and min(
                        abs(90 - angle_023), 180 - angle_023) > 10) and min02 > min20 + 1:
                        conti = False
                    elif 80 <= angle <= 100 and not i201 and not i012 and min02 > min20 + 1:
                        conti = False

                    elif 80 < angle__3 < 100 and min01 >= min_0 and min12 >= min23 and not h0 and not h1 and (
                            angle__2 > 10 and angle_03 > 10 or angle < 60):  # and dis__0>dis_01 and dis_23>dis_12
                        conti = 3
                    elif angle__3 <= 15 and (dis__0 + dis_23) >= (dis_01 + dis_12):
                        conti = 1 if intersect_12 or intersect_01 else 2
                    elif 75 <= angle__2 and intersect_12 or 75 <= angle_03 and intersect_01:
                        conti = 1
                    elif i201:
                        conti = 1
                    elif i012:
                        conti = 1
                    elif beter02:
                        if not skip:
                            conti = 2
                    elif p201:
                        conti = 5
                    elif p012:
                        conti = 4
                    elif (area_judg(area_1, contour_area, -1,
                                    seg_type) or dis_01 < 5 or dis_12 < 5 or dis_01 + dis_12 < 14) and min20 >= 15:
                        conti = 2
                    elif intersect_12 and area_judg(area_c_12, contour_area, -1, seg_type) and min01 >= 15:
                        conti = 1
                    elif intersect_01 and area_judg(area_c_01, contour_area, -1, seg_type) and min12 >= 15:
                        conti = 1

                    elif 150 <= angle <= 165 and (
                            min20 > 5 and min02 <= min12 * proportion_12 + min01 * proportion_01 - 1):
                        conti = 2
                    elif 70 < angle < 150 and min20 > 15 and (diff_015 or diff_125):
                        conti = 2
                    elif angle < 70:
                        conti = 1

                    if conti == 5:
                        if area_judg(area_2, contour_area, 2, seg_type):
                            approx[i] = [x_2, y_2]
                            if step_show: show_result(appro, approx, x1, y1, True)
                            continue
                    elif conti == 4:
                        if area_judg(area_0, contour_area, 2, seg_type):
                            approx[i] = [x_0, y_0]
                            if step_show: show_result(appro, approx, x1, y1, True)
                            continue
                    elif conti == 3:
                        x_0, y_0 = Drop_point(A_, B_, C_, x2, y2)
                        x_2, y_2 = Drop_point(A2, B2, C2, x0, y0)
                        if 85 <= angle__3 <= 95:
                            x_02, y_02 = findIntersection(x_, y_, x0, y0, x2, y2, x3, y3)
                        elif lineup(x_, y_, x0, y0, x_0, y_0):
                            x_02, y_02 = x_0, y_0
                        elif lineup(x2, y2, x3, y3, x_2, y_2):
                            x_02, y_02 = x_2, y_2
                        elif min_0 <= min23:
                            x_02, y_02 = x_0, y_0
                        else:
                            x_02, y_02 = x_2, y_2
                        try:
                            area_02 = union([[x0, y0], [x1, y1], [x2, y2]], [[x0, y0], [x2, y2], [x_02, y_02]])
                        except Exception:
                            continue
                        if area_judg(area_02, contour_area, 2, seg_type):
                            if 85 <= angle__3 <= 95:
                                approx[(i + 1) % len(approx)] = [x_02, y_02]
                                approx.pop(i)
                                approx.pop(i - 1)
                                jump += 1
                            elif lineup(x_, y_, x0, y0, x_0, y_0):
                                approx[(i + 1) % len(approx)] = [x_02, y_02]
                                approx.pop(i)
                            elif lineup(x2, y2, x3, y3, x_2, y_2):
                                approx[i - 1] = [x_02, y_02]
                                approx.pop(i)
                            elif min_0 <= min23:
                                approx[(i + 1) % len(approx)] = [x_02, y_02]
                                approx.pop(i)
                            else:
                                approx[i - 1] = [x_02, y_02]
                                approx.pop(i)
                            if step_show: show_result(appro, approx, x1, y1, True)
                            continue
                    elif conti == 2 and area_judg(area_1, contour_area, 3, seg_type) and (
                            5 < angle_0x < 85 and 5 < angle_1x < 85 or min02x <= min(min0x, min1x)):
                        approx.pop(i)
                        if step_show: show_result(appro, approx, x1, y1, True)
                        continue
                    elif conti == 1:
                        if intersect_12:
                            if math.sqrt((y2 - c_y_12) ** 2 + (x2 - c_x_12) ** 2) > 6 and area_judg(area_c_12,
                                                                                                    contour_area, 3,
                                                                                                    seg_type):
                                approx[(i + 1) % len(approx)] = [c_x_12, c_y_12]
                                approx.pop(i)
                                if step_show: show_result(appro, approx, x1, y1, True)
                                continue
                            elif area_judg(area_1, contour_area, 3, seg_type):
                                approx.pop(i)
                                if step_show: show_result(appro, approx, x1, y1, True)
                                continue
                        elif intersect_01:
                            if math.sqrt((y0 - c_y_01) ** 2 + (x0 - c_x_01) ** 2) > 6 and area_judg(area_c_01,
                                                                                                    contour_area, 3,
                                                                                                    seg_type):
                                approx[i - 1] = [c_x_01, c_y_01]
                                approx.pop(i)
                                if step_show: show_result(appro, approx, x1, y1, True)
                                continue
                            elif area_judg(area_1, contour_area, 3, seg_type):
                                approx.pop(i)
                                if step_show: show_result(appro, approx, x1, y1, True)
                                continue
                        elif dis_01 + 5 < dis_12:
                            sin, cos = (y2 - y1) / dis_12, (x2 - x1) / dis_12
                            c_x, c_y = x1 + dis_01 * cos, y1 + dis_01 * sin
                            area_c = polygon_area([[x0, y0], [x1, y1], [c_x, c_y]])
                            if not h0 and area_judg(area_c, contour_area, 2, seg_type):
                                approx[i] = [c_x, c_y]
                                if step_show: show_result(appro, approx, x1, y1, True)
                                continue
                        elif dis_12 + 5 < dis_01:
                            sin, cos = (y0 - y1) / dis_01, (x0 - x1) / dis_01
                            c_x, c_y = x1 + dis_12 * cos, y1 + dis_12 * sin
                            area_c = polygon_area([[x2, y2], [x1, y1], [c_x, c_y]])
                            if not h1 and area_judg(area_c, contour_area, 2, seg_type):
                                approx[i] = [c_x, c_y]
                                if step_show: show_result(appro, approx, x1, y1, True)
                                continue
                        elif area_judg(area_1, contour_area, 2, seg_type):
                            approx.pop(i)
                            if step_show: show_result(appro, approx, x1, y1, True)
                            continue

                elif rayCasting([x2, y2], app_2) and not rayCasting([x3, y3], app_3) and not rayCasting([x0, y0],
                                                                                                        app_0) and angle_2 <= 165:
                    conti, cont = False, False
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if angle__4 < 15 and angle__2 <= 45:
                        A_1, B_1, C_1 = two_points_line(x_, y_, x0, y0, cx, cy) if min_0 <= min(90 - angle_34,
                                                                                                angle_34 - 0) else two_points_line(
                            x3, y3, x4, y4, cx, cy)
                    elif angle__2 < 15 or angle__2 > 75 or angle_14 < 15 or angle_14 > 75:
                        A_1, B_1, C_1 = A1, B1, C1
                    elif angle_12 < 15:
                        A_1, B_1, C_1 = two_points_line(startx, starty, endx, endy, cx, cy)
                    elif angle_12 > 75:
                        A_1, B_1, C_1 = two_points_vline(startx, starty, endx, endy, cx, cy)
                    else:
                        A_1 = None
                    if A_1:
                        x_1, y_1 = Drop_point(A_1, B_1, C_1, x0, y0)
                        x_f1, y_f1 = findIntersection(cx, cy, x_1, y_1, x0, y0, x1, y1)
                        x_y1, y_y1 = 1.5 * x1 - 0.5 * x0, 1.5 * y1 - 0.5 * y0
                        x_2, y_2 = Drop_point(A_1, B_1, C_1, x3, y3)
                        x_f2, y_f2 = findIntersection(cx, cy, x_2, y_2, x2, y2, x3, y3)
                        x_y2, y_y2 = 1.5 * x2 - 0.5 * x3, 1.5 * y2 - 0.5 * y3
                        angle_0_1, angle_2_3 = cal_angle((x_1 - x0, y_1 - y0), vec_se), cal_angle((x3 - x_2, y3 - y_2),
                                                                                                  vec_se)
                        area_x_0, area_x_3 = union([[x0, y0], [x_1, y_1], [cx, cy]],
                                                   [[x0, y0], [x1, y1], [cx, cy]]), union(
                            [[x3, y3], [x_2, y_2], [cx, cy]], [[x3, y3], [x2, y2], [cx, cy]])

                        if 80 <= angle_0 <= 100 and x_f1 and lineup(x0, y0, x_y1, y_y1, x_f1, y_f1):
                            approx[i] = [x_f1, y_f1]
                            conti = True
                        elif dis__0 >= dis_01 and (45 < angle_0 < 135 or 80 <= angle__2):
                            if angle__2 < 15:
                                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                                x_0, y_0 = Drop_point(A_, B_, C_, cx, cy)
                                x_1, y_1 = Drop_point(A1, B1, C1, cx, cy)
                                if area_judg(polygon_area([[x0, y0], [x_0, y_0], [cx, cy]]) + polygon_area(
                                        [[x1, y1], [x_1, y_1], [cx, cy]]), contour_area, 2, seg_type):
                                    approx[(i + 1) % len(approx)] = [x_0, y_0]
                                    approx[i] = [x_1, y_1]
                                    conti = True
                            elif 75 < angle__2 and lineup(x1, y1, x2, y2, x_1, y_1) and area_judg(
                                    polygon_area([[x1, y1], [x_1, y_1], [x0, y0]]), contour_area, 2, seg_type):
                                approx[i] = [x_1, y_1]
                                conti = True
                            elif [x_, y_] == [x0, y0] or [x0, y0] == [x_1, y_1]:
                                approx[i] = [x_1, y_1]
                                approx.remove([x0, y0])
                                conti = True
                                if i + 1 == len(approx): cont = True
                            elif abs(cal_ang((x_, y_), (x0, y0), (x_1, y_1)) - 90) <= abs(angle_0 - 90) and cal_ang(
                                    (x_, y_), (x0, y0), (x_1, y_1)) >= 60 and min(90 - angle_0_1,
                                                                                  angle_0_1 - 0) <= min01 and area_judg(
                                area_x_0, contour_area, 2, seg_type):
                                approx[i] = [x_1, y_1]
                                conti = True
                            else:
                                try:
                                    x_1, y_1 = Drop_point(A1, B1, C1, x0, y0)
                                    c_x, c_y = (x_1 + x1) / 2, (y_1 + y1) / 2
                                    if abs(cal_ang((x_, y_), (x0, y0), (c_x, c_y)) - 90) <= abs(angle_0 - 90) and abs(
                                            cal_ang((x0, y0), (c_x, c_y), (x2, y2)) - 90) <= abs(angle - 90):
                                        approx[i] = [c_x, c_y]
                                        conti = True
                                except:
                                    pass
                        if 80 <= angle_3 <= 100 and x_f2 and lineup(x3, y3, x_y2, y_y2, x_f2, y_f2):
                            approx[i - 1] = [x_f2, y_f2]
                            cont = True
                        elif dis_34 >= dis_23 and (45 < angle_3 < 135 or 80 <= angle_14):
                            if angle_14 < 15:
                                cx, cy = (x2 + x3) / 2, (y2 + y3) / 2
                                x_2, y_2 = Drop_point(A1, B1, C1, cx, cy)
                                x_3, y_3 = Drop_point(A3, B3, C3, cx, cy)
                                if area_judg(polygon_area([[x2, y2], [x_2, y_2], [cx, cy]]) + polygon_area(
                                        [[x3, y3], [x_3, y_3], [cx, cy]]), contour_area, 2, seg_type):
                                    approx[i - 1] = [x_2, y_2]
                                    approx[i - 2] = [x_3, y_3]
                                    cont = True
                            elif 75 < angle_14 and lineup(x1, y1, x2, y2, x_2, y_2) and area_judg(
                                    polygon_area([[x2, y2], [x_2, y_2], [x3, y3]]), contour_area, 2, seg_type):
                                approx[i - 1] = [x_2, y_2]
                                cont = True
                            elif [x4, y4] == [x3, y3] or [x3, y3] == [x_2, y_2]:
                                approx[i - 1] = [x_2, y_2]
                                approx.remove([x3, y3])
                                if i - 2 >= 0: cont = True
                            elif abs(cal_ang((x4, y4), (x3, y3), (x_2, y_2)) - 90) <= abs(angle_3 - 90) and cal_ang(
                                    (x4, y4), (x3, y3), (x_2, y_2)) >= 60 and min(90 - angle_2_3,
                                                                                  angle_2_3 - 0) <= min23 and area_judg(
                                area_x_3, contour_area, 2, seg_type):
                                approx[i - 1] = [x_2, y_2]
                                cont = True
                            else:
                                try:
                                    x_2, y_2 = Drop_point(A1, B1, C1, x3, y3)
                                    c_x, c_y = (x_2 + x2) / 2, (y_2 + y2) / 2
                                    if abs(cal_ang((x4, y4), (x3, y3), (c_x, c_y)) - 90) <= abs(angle_3 - 90) and abs(
                                            cal_ang((x3, y3), (c_x, c_y), (x1, y1)) - 90) <= abs(angle_2 - 90):
                                        approx[i - 1] = [c_x, c_y]
                                        cont = True
                                except:
                                    pass
                        if cont:
                            jump += 1
                            if step_show: show_result(appro, approx, x1, y1, True)
                        if conti == 1:
                            if step_show: show_result(appro, approx, x1, y1, True)
                            continue
                elif rayCasting([x0, y0], app_0) and rayCasting([x2, y2], app_2) and not rayCasting([x_, y_],
                                                                                                    app__) and not rayCasting(
                    [x3, y3], app_3):
                    if angle__3 >= 170:
                        conti = False
                        if dis__0 >= 0.8 * dis_01 and min_0 <= min01:
                            x_0, y_0 = Drop_point(A_, B_, C_, x1, y1)
                            if area_judg(polygon_area([[x0, y0], [x_0, y_0], [x1, y1]]), contour_area, 1, seg_type):
                                approx[(i + 1) % len(approx)] = [x_0, y_0]
                                conti += 1
                        if dis_23 >= 0.8 * dis_12 and min23 <= min12:
                            x_2, y_2 = Drop_point(A2, B2, C2, x1, y1)
                            if area_judg(polygon_area([[x1, y1], [x_2, y_2], [x2, y2]]), contour_area, 1, seg_type):
                                approx[i - 1] = [x_2, y_2]
                                conti += 1
                        if conti:
                            if conti == 2: approx.pop(i)
                            continue
                    elif angle_1_4 <= 10:
                        conti = False
                        Ac, Bc, Cc = two_points_line(x1_, y1_, x_, y_, x1, y1) if min1__ <= min34 else two_points_line(
                            x3, y3, x4, y4, x1, y1)
                        if min1__ <= min(min_0, min01) and dis_1__ >= 0.8 * (dis__0 + dis_01):
                            cx, cy = (x_ + x0) / 2, (y_ + y0) / 2
                            x__, y__ = Drop_point(A1_, B1_, C1_, cx, cy)
                            x_0, y_0 = Drop_point(Ac, Bc, Cc, cx, cy)
                            area_c = union([[x_, y_], [x0, y0], [x1, y1]], [[x__, y__], [x_0, y_0], [x1, y1]])
                            if area_judg(area_c, contour_area, 2, seg_type):
                                approx[(i + 2) % len(approx)] = [x__, y__]
                                approx[(i + 1) % len(approx)] = [x_0, y_0]
                                conti += 1
                        if min34 <= min(min12, min23) and dis_34 >= 0.8 * (dis_12 + dis_23):
                            cx, cy = (x2 + x3) / 2, (y2 + y3) / 2
                            x_3, y_3 = Drop_point(A3, B3, C3, cx, cy)
                            x_2, y_2 = Drop_point(Ac, Bc, Cc, cx, cy)
                            area_c = union([[x1, y1], [x2, y2], [x3, y3]], [[x1, y1], [x_2, y_2], [x_3, y_3]])
                            if area_judg(area_c, contour_area, 2, seg_type):
                                approx[i - 2] = [x_3, y_3]
                                approx[i - 1] = [x_2, y_2]
                                conti += 1
                        if conti:
                            if conti == 2: approx.pop(i)
                            continue
            #########################################################################################################################################处理大钝角,连成一条线
            if angle >= 150:
                try:
                    conti = None
                    cx_01, cy_01 = (x0 + x1) / 2, (y0 + y1) / 2
                    cx_12, cy_12 = (x2 + x1) / 2, (y2 + y1) / 2
                    Ac, Bc, Cc = two_points_line(cx_01, cy_01, cx_12, cy_12, cx_01, cy_01)
                    x_0, y_0 = Drop_point(Ac, Bc, Cc, x0, y0)
                    x_2, y_2 = Drop_point(Ac, Bc, Cc, x2, y2)
                    angle___0_2, angle__0_23 = cal_ang((x_, y_), (x_0, y_0), (x_2, y_2)), cal_ang((x_0, y_0),
                                                                                                  (x_2, y_2), (x3, y3))
                    if angle >= 167:
                        conti = 1
                    elif area_judg(area_1, contour_area, 4, seg_type) and (min0x > 5 and min1x > 5 or min02x <= 5):
                        if angle_03 <= 10 and d1_23 > 7 and min(dis_01, dis_23) >= 8 and min30 > min(min01,
                                                                                                     min23) and min02 > min20:  # 四点一线没解决
                            conti = False
                        elif angle__2 <= 10 and d0_12 > 7 and min(dis__0, dis_12) >= 8 and min2_ > min(min_0,
                                                                                                       min12) and min02 > min20:
                            conti = False
                        elif angle_03 >= 80 and min(dis_01, dis_23) >= 8 and min02 > min20 and abs(
                                90 - angle_023) > 90 - angle_03:
                            conti = False
                        elif angle__2 >= 80 and min(dis__0, dis_12) >= 8 and min02 > min20 and abs(
                                90 - angle__02) > 90 - angle__2:
                            conti = False
                        elif min20 <= 10:
                            if max(dis_01, dis_12) / min(dis_01, dis_12) <= 2.5 and min02 <= min20 + 1 and (
                                    min(min0x, min1x) > 5 or min02x <= min(min0x, min1x) + 1):
                                conti = 1
                            elif min02 <= min20 and (min(min0x, min1x) > 5 or min02x <= min(min0x, min1x)):
                                conti = 1
                        elif min20 > 10 and min02 < min12 * proportion_12 + min01 * proportion_01 and max(dis_01,
                                                                                                          dis_12) / min(
                            dis_01, dis_12) <= 3:
                            conti = 1
                        elif rayCasting([x1, y1], appr) and min(abs(90 - angle__02), abs(180 - angle__02)) <= min(
                                abs(90 - angle_0), abs(180 - angle_0), 15) and min(abs(90 - angle_023),
                                                                                   abs(180 - angle_023)) <= min(
                            abs(90 - angle_2), abs(180 - angle_2), 15) \
                                or not rayCasting([x1, y1], appr) and min(abs(90 - angle___0_2),
                                                                          abs(180 - angle___0_2)) <= min(
                            abs(90 - angle_0), abs(180 - angle_0), 15) and min(abs(90 - angle__0_23),
                                                                               abs(180 - angle__0_23)) <= min(
                            abs(90 - angle_2), abs(180 - angle_2), 15):
                            conti = 1
                    if conti == 1:
                        if not rayCasting([x1, y1], appr):
                            approx[(i + 1) % len(approx)] = [x_0, y_0]
                            approx[i - 1] = [x_2, y_2]
                        approx.pop(i)
                        if step_show: show_result(appro, approx, x1, y1, True)
                        if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx, miny, True,
                                                                                              seg_type)
                        continue
                except:
                    pass
            ####################################################################################################################################处理小锐角
            if angle < 18:
                c_x_12, c_y_12, c_x_01, c_y_01, area_c_12, area_c_01, intersect_12, intersect_01 = intersect(x_, y_, x0,
                                                                                                             y0, x1, y1,
                                                                                                             x2, y2, x3,
                                                                                                             y3, True)
                if intersect_12 and area_judg(area_c_12, contour_area, 1, seg_type) and math.sqrt(
                        (y2 - c_y_12) ** 2 + (x2 - c_x_12) ** 2) >= 5:
                    approx[(i + 1) % len(approx)] = [c_x_12, c_y_12]
                    approx.pop(i)
                elif intersect_01 and area_judg(area_c_01, contour_area, 1, seg_type) and math.sqrt(
                        (y0 - c_y_01) ** 2 + (x0 - c_x_01) ** 2) >= 5:
                    approx[i - 1] = [c_x_01, c_y_01]
                    approx.pop(i)
                elif area_judg(area_1, contour_area, 1, seg_type):
                    approx.pop(i)
                if step_show: show_result(appro, approx, x1, y1, True)
            #############################################直角规整
            elif 83 <= angle <= 97 and min20 <= 10:
                if round(angle) == 90:
                    pass
                elif (min01 <= min12 and dis_01 >= 0.8 * dis_12) and 5 < angle_1x < 85:
                    c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
                    x_1, y_1 = Drop_point(A0, B0, C0, c_x, c_y)
                    Ac, Bc, Cc = two_points_line(x_1, y_1, c_x, c_y, c_x, c_y)
                    x_2, y_2 = Drop_point(Ac, Bc, Cc, x2, y2)
                    area_c = polygon_area([[x1, y1], [x_1, y_1], [c_x, c_y]]) + polygon_area(
                        [[x2, y2], [x_2, y_2], [c_x, c_y]])
                    if area_judg(area_c, contour_area, 0, seg_type):
                        approx[i] = [x_1, y_1]
                        approx[i - 1] = [x_2, y_2]
                        if step_show: show_result(appro, approx, x1, y1, True)
                elif (min12 < min01 and dis_12 >= 0.8 * dis_01) and 5 < angle_0x < 85:
                    c_x, c_y = (x1 + x0) / 2, (y1 + y0) / 2
                    x_1, y_1 = Drop_point(A1, B1, C1, c_x, c_y)
                    Ac, Bc, Cc = two_points_line(x_1, y_1, c_x, c_y, c_x, c_y)
                    x_0, y_0 = Drop_point(Ac, Bc, Cc, x0, y0)
                    area_c = polygon_area([[x1, y1], [x_1, y_1], [c_x, c_y]]) + polygon_area(
                        [[x0, y0], [x_0, y_0], [c_x, c_y]])
                    if area_judg(area_c, contour_area, 0, seg_type):
                        approx[i] = [x_1, y_1]
                        approx[(i + 1) % len(approx)] = [x_0, y_0]
                        if step_show: show_result(appro, approx, x1, y1, True)
            ############################################钝角规整
            elif (110 < angle < 170 and min20 <= 10 or (
                    dis_01 / dis_12 > 3 or dis_12 / dis_01 > 3) and angle >= 150) and seg_type == 'bfp':
                if min01 <= min12 and dis_01 >= 0.8 * dis_12 and (
                        min1x > 5 or dis_01 / dis_12 >= 3) or dis_01 / dis_12 >= 3 and angle >= 150 and min1x > 5:
                    c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
                    x_12, y_12 = findIntersection(x0, y0, x1, y1, x2, y2, x3, y3)
                    if not rayCasting([x1, y1], appr) and not rayCasting([x2, y2], app_2):
                        if angle_03 <= 10 and dis_23 >= 8:
                            continue
                        elif angle_03 > 85 and dis_23 >= 8:
                            x_1, y_1 = x_12, y_12
                        elif dis_01 <= 1.2 * dis_12:
                            x_1, y_1 = Drop_point(A0, B0, C0, c_x, c_y)
                        else:
                            x_1, y_1 = Drop_point(A0, B0, C0, x2, y2)
                        if x_12 and lineup(x1, y1, x_1, y_1, x_12, y_12) and lineup(x2, y2, x3, y3, x_12, y_12):
                            continue
                        area_c = polygon_area([[x1, y1], [x_1, y_1], [x2, y2]])
                        if area_judg(area_c, contour_area, 2, seg_type):
                            if angle <= 160 and not h1 or dis_01 >= 5 * dis_12 or 160 < angle < 170:
                                approx[i] = [x_1, y_1]
                                if step_show: show_result(appro, approx, x1, y1, True)
                                if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx,
                                                                                                      miny, True,
                                                                                                      seg_type)
                    elif rayCasting([x1, y1], appr):
                        if angle_03 <= 13 and angle < 160 and d1_23 > 5:
                            x_1, y_1 = Drop_point(A0, B0, C0, c_x, c_y)
                            x_2, y_2 = Drop_point(A2, B2, C2, c_x, c_y)
                            area_c = 2 * polygon_area([[x1, y1], [x_1, y_1], [c_x, c_y]])
                            if area_judg(area_c, contour_area, 0, seg_type):
                                approx[i] = [x_1, y_1]
                                approx[i - 1] = [x_2, y_2]
                                if step_show: show_result(appro, approx, x1, y1, True)
                        else:
                            if angle >= 160 or dis_01 > 1.5 * dis_12:
                                c_x, c_y = x2, y2
                            x_1, y_1 = Drop_point(A0, B0, C0, c_x, c_y)
                            angle__123, dis__12 = cal_ang((x_1, y_1), (x2, y2), (x3, y3)), math.sqrt(
                                (y2 - y_1) ** 2 + (x2 - x_1) ** 2)
                            area_c = polygon_area([[x1, y1], [x_1, y_1], [x2, y2]])
                            if area_judg(area_c, contour_area, 0, seg_type) and dis__12 >= 7 and (
                                    rayCasting([x2, y2], app_2) or min(abs(90 - angle__123),
                                                                       abs(180 - angle__123)) <= min(abs(90 - angle_2),
                                                                                                     abs(180 - angle_2)) and angle__123 >= 75):
                                approx[i] = [x_1, y_1]
                                if step_show: show_result(appro, approx, x1, y1, True)

                elif min12 < min01 and dis_12 >= 0.8 * dis_01 and (
                        min0x > 5 or dis_12 / dis_01 >= 3) or dis_12 / dis_01 >= 3 and angle >= 150 and 5 < angle_0x < 85:
                    c_x, c_y = (x1 + x0) / 2, (y1 + y0) / 2
                    x_12, y_12 = findIntersection(x_, y_, x0, y0, x1, y1, x2, y2)
                    if not rayCasting([x1, y1], appr) and not rayCasting([x0, y0], app_0):
                        if angle__2 <= 10 and dis__0 >= 8:
                            continue
                        elif angle__2 > 85 and dis__0 >= 8:
                            x_1, y_1 = x_12, y_12
                        elif dis_12 <= 1.2 * dis_01:
                            x_1, y_1 = Drop_point(A1, B1, C1, c_x, c_y)
                        else:
                            x_1, y_1 = Drop_point(A1, B1, C1, x0, y0)
                        if x_12 and lineup(x1, y1, x_1, y_1, x_12, y_12) and lineup(x_, y_, x0, y0, x_12, y_12):
                            continue
                        area_c = polygon_area([[x1, y1], [x_1, y_1], [x0, y0]])
                        if area_judg(area_c, contour_area, 2, seg_type):
                            if angle <= 160 and not h0 or dis_12 >= 5 * dis_01 or 160 < angle < 170:
                                approx[i] = [x_1, y_1]
                                if step_show: show_result(appro, approx, x1, y1, True)
                                if hierarchy[0, c, 3] == -1: startx, starty, endx, endy, vec_se = zhu(approx, minx,
                                                                                                      miny, True,
                                                                                                      seg_type)
                    elif rayCasting([x1, y1], appr):
                        if angle__2 <= 13 and angle < 160 and d0_12 > 5:
                            x_0, y_0 = Drop_point(A_, B_, C_, c_x, c_y)
                            x_1, y_1 = Drop_point(A1, B1, C1, c_x, c_y)
                            area_c = 2 * polygon_area([[x1, y1], [x_1, y_1], [c_x, c_y]])
                            if area_judg(area_c, contour_area, 0, seg_type):
                                approx[i] = [x_1, y_1]
                                approx[(i + 1) % len(approx)] = [x_0, y_0]
                                if step_show: show_result(appro, approx, x1, y1, True)
                        else:
                            if angle >= 160 or dis_12 > 1.5 * dis_01:
                                c_x, c_y = x0, y0
                            x_1, y_1 = Drop_point(A1, B1, C1, c_x, c_y)
                            angle__10_, dis__10 = cal_ang((x_1, y_1), (x0, y0), (x_, y_)), math.sqrt(
                                (y0 - y_1) ** 2 + (x0 - x_1) ** 2)
                            area_c = polygon_area([[x1, y1], [x_1, y_1], [x0, y0]])
                            if area_judg(area_c, contour_area, 0, seg_type) and dis__10 >= 7 and (
                                    rayCasting([x0, y0], app_0) or min(abs(90 - angle__10_),
                                                                       abs(180 - angle__10_)) <= min(abs(90 - angle_0),
                                                                                                     abs(180 - angle_0)) and angle__10_ >= 75):
                                approx[i] = [x_1, y_1]
                                if step_show: show_result(appro, approx, x1, y1, True)
            ###########################################锐角规整
            elif angle < 75 and min20 <= 10:
                if min01 <= min12 and (angle <= 65 or angle > 65 and not h1) and (
                        angle <= 45 or min1x > 5 or dis_01 >= 2.5 * dis_12):
                    conti = False
                    if not rayCasting([x1, y1], appr) and (angle <= 45 or dis_01 >= 0.5 * dis_12):
                        if angle_03 <= 10:
                            x_2, y_2 = Drop_point(A2, B2, C2, x1, y1)
                            area_2 = polygon_area([[x1, y1], [x_2, y_2], [x2, y2]])
                            if angle_2 > 90 and area_judg(area_2, contour_area, 2, seg_type):
                                approx[i - 1] = [x_2, y_2]
                                if step_show: show_result(appro, approx, x1, y1, True)
                            elif angle_2 < 90:
                                c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
                                x_2, y_2 = Drop_point(A2, B2, C2, c_x, c_y)
                                x_1, y_1 = Drop_point(A0, B0, C0, c_x, c_y)
                                area_c = polygon_area([[x1, y1], [c_x, c_y], [x_1, y_1]]) + polygon_area(
                                    [[x2, y2], [c_x, c_y], [x_2, y_2]])
                                if area_judg(area_c, contour_area, 1, seg_type):
                                    approx[i - 1] = [x_2, y_2]
                                    approx[i] = [x_1, y_1]
                                    if step_show: show_result(appro, approx, x1, y1, True)
                        else:
                            Ac, Bc, Cc = two_points_line(x0, y0, x1, y1, x2, y2)
                            x_1_d, y_1_d = Drop_point(Ac, Bc, Cc, x1, y1)
                            area_x_1 = polygon_area([[x1, y1], [x_1_d, y_1_d], [x2, y2]])
                            if area_judg(area_x_1, contour_area, 2,
                                         seg_type) and angle > 45 or angle <= 45:  # area_judg(area_x_1,contour_area,6,seg_type) and angle<=45 or
                                conti = 1
                            elif area_judg(area_x_1, contour_area, 4, seg_type):
                                c_x, c_y = x1 + (x2 - x1) / 3, y1 + (y2 - y1) / 3
                                x_1_d, y_1_d = Drop_point(Ac, Bc, Cc, c_x, c_y)
                                conti = 1
                            elif area_judg(area_x_1, contour_area, 5, seg_type):
                                c_x, c_y = (x2 + x1) / 2, (y2 + y1) / 2
                                x_1_d, y_1_d = Drop_point(Ac, Bc, Cc, c_x, c_y)
                                conti = 1
                    elif rayCasting([x1, y1], appr):
                        x_1_d, y_1_d = Drop_point(A0, B0, C0, x2, y2)
                        area_x_1 = polygon_area([[x1, y1], [x_1_d, y_1_d], [x2, y2]])
                        if area_judg(area_x_1, contour_area, 2,
                                     seg_type) and angle > 45 or angle <= 45:  # area_judg(area_x_1,contour_area,5,seg_type) and angle<=45 or
                            conti = 2
                        elif area_judg(area_x_1, contour_area, 4, seg_type):
                            c_x, c_y = x2 + (x1 - x2) / 3, y2 + (y1 - y2) / 3
                            x_1_d, y_1_d = Drop_point(A0, B0, C0, c_x, c_y)
                            conti = 2
                        elif area_judg(area_x_1, contour_area, 5, seg_type):
                            c_x, c_y = (x1 + x2) / 2, (y1 + y2) / 2
                            x_1_d, y_1_d = Drop_point(A0, B0, C0, c_x, c_y)
                            conti = 2
                    if conti == 1:
                        area_3 = polygon_area([[x_1_d, y_1_d], [x2, y2], [x3, y3]])
                        dis_d2 = math.sqrt((y_1_d - y2) ** 2 + (x_1_d - x2) ** 2)
                        if area_judg(area_3, contour_area, -3, seg_type) and dis_d2 <= 5 and 15 <= angle_23 <= 75:
                            approx[i - 1] = [x_1_d, y_1_d]
                        elif rayCasting([x2, y2], app_2) and (
                                [x2, y2] != approx[-1] or cal_ang((x_1_d, y_1_d), (x2, y2), (x3, y3)) <= 45):
                            continue
                        else:
                            approx.insert(i, [x_1_d, y_1_d])
                        if step_show: show_result(appro, approx, x1, y1, True)
                    elif conti == 2:
                        if cal_ang((x1, y1), (x2, y2), (x_1_d, y_1_d)) > angle_2 and rayCasting([x2, y2], app_2):
                            continue
                        elif lineup(x0, y0, x1, y1, x_1_d, y_1_d) and math.sqrt(
                                (y0 - y_1_d) ** 2 + (x0 - x_1_d) ** 2) >= 4:
                            approx[i] = [x_1_d, y_1_d]
                        elif angle <= 45 or math.sqrt((y0 - y_1_d) ** 2 + (x0 - x_1_d) ** 2) < 4:
                            approx.pop(i)
                        if step_show: show_result(appro, approx, x1, y1, True)

                elif min12 < min01 and (angle <= 65 or angle > 65 and not h0) and (
                        angle <= 45 or min0x > 5 or dis_12 >= 2.5 * dis_01):
                    conti = False
                    if not rayCasting([x1, y1], appr) and (angle <= 45 or dis_12 >= 0.5 * dis_01):
                        if angle__2 <= 10:
                            x_0, y_0 = Drop_point(A_, B_, C_, x1, y1)
                            area_0 = polygon_area([[x1, y1], [x_0, y_0], [x0, y0]])
                            if angle_0 > 90 and area_judg(area_0, contour_area, 2, seg_type):
                                approx[(i + 1) % len(approx)] = [x_0, y_0]
                                if step_show: show_result(appro, approx, x1, y1, True)
                            elif angle_0 < 90:
                                c_x, c_y = (x1 + x0) / 2, (y1 + y0) / 2
                                x_0, y_0 = Drop_point(A_, B_, C_, c_x, c_y)
                                x_1, y_1 = Drop_point(A1, B1, C1, c_x, c_y)
                                area_c = polygon_area([[x1, y1], [c_x, c_y], [x_1, y_1]]) + polygon_area(
                                    [[x0, y0], [c_x, c_y], [x_0, y_0]])
                                if area_judg(area_c, contour_area, 1, seg_type):
                                    approx[(i + 1) % len(approx)] = [x_0, y_0]
                                    approx[i] = [x_1, y_1]
                                    if step_show: show_result(appro, approx, x1, y1, True)
                        else:
                            Ac, Bc, Cc = two_points_line(x1, y1, x2, y2, x0, y0)
                            x_1_u, y_1_u = Drop_point(Ac, Bc, Cc, x1, y1)
                            area_x_1 = polygon_area([[x1, y1], [x_1_u, y_1_u], [x0, y0]])
                            if area_judg(area_x_1, contour_area, 2,
                                         seg_type) and angle > 45 or angle <= 45:  # or area_judg(area_x_1,contour_area,6,seg_type) and angle<=45
                                conti = 1
                            elif area_judg(area_x_1, contour_area, 4, seg_type):
                                c_x, c_y = x1 + (x0 - x1) / 3, y1 + (y0 - y1) / 3
                                x_1_u, y_1_u = Drop_point(Ac, Bc, Cc, c_x, c_y)
                                conti = 1
                            elif area_judg(area_x_1, contour_area, 5, seg_type):
                                c_x, c_y = (x0 + x1) / 2, (y0 + y1) / 2
                                x_1_u, y_1_u = Drop_point(Ac, Bc, Cc, c_x, c_y)
                                conti = 1
                    elif rayCasting([x1, y1], appr):
                        x_1_u, y_1_u = Drop_point(A1, B1, C1, x0, y0)
                        area_x_1 = polygon_area([[x1, y1], [x_1_u, y_1_u], [x0, y0]])
                        if area_judg(area_x_1, contour_area, 2,
                                     seg_type) and angle > 45 or angle <= 45:  # area_judg(area_x_1,contour_area,5,seg_type) and angle<=45
                            conti = 2
                        elif area_judg(area_x_1, contour_area, 4, seg_type):
                            c_x, c_y = x0 + (x1 - x0) / 3, y0 + (y1 - y0) / 3
                            x_1_u, y_1_u = Drop_point(A1, B1, C1, c_x, c_y)
                            conti = 2
                        elif area_judg(area_x_1, contour_area, 5, seg_type):
                            c_x, c_y = (x1 + x0) / 2, (y1 + y0) / 2
                            x_1_u, y_1_u = Drop_point(A1, B1, C1, c_x, c_y)
                            conti = 2
                    if conti == 1:
                        area__ = polygon_area([[x_1_u, y_1_u], [x0, y0], [x_, y_]])
                        dis_u0 = math.sqrt((y_1_u - y0) ** 2 + (x_1_u - x0) ** 2)
                        if area_judg(area__, contour_area, -3, seg_type) and dis_u0 <= 5 and 15 <= angle__0 <= 75:
                            approx[(i + 1) % len(approx)] = [x_1_u, y_1_u]
                        elif rayCasting([x0, y0], app_0) and (
                                (i + 1) % len(approx) == 0 or cal_ang((x_1_u, y_1_u), (x0, y0), (x_, y_)) <= 45):
                            continue
                        else:
                            approx.insert(i + 1, [x_1_u, y_1_u])
                        if step_show: show_result(appro, approx, x1, y1, True)
                    elif conti == 2:
                        if cal_angle((x1, y1), (x0, y0), (x_1_u, y_1_u)) > angle_0 and rayCasting([x0, y0], app_0):
                            continue
                        elif lineup(x1, y1, x2, y2, x_1_u, y_1_u) and math.sqrt(
                                (y2 - y_1_u) ** 2 + (x2 - x_1_u) ** 2) >= 4:
                            approx[i] = [x_1_u, y_1_u]
                        elif angle <= 45 or math.sqrt((y2 - y_1_u) ** 2 + (x2 - x_1_u) ** 2) < 4:
                            approx.pop(i)
                        if step_show: show_result(appro, approx, x1, y1, True)
            #############################################################################################锐角扩展
            elif 10 < angle_01 < 80 and 10 < angle_12 < 80 and angle <= 80:
                conti = False
                if 73 < angle_0 < 107 and 73 < angle_2 < 107 and angle > 60 or angle_03 <= 10 or angle_03 >= 80 or angle__2 <= 10 or angle__2 >= 80:
                    continue
                vec_c = None
                if angle__3 <= 7 or angle__3 >= 173:
                    if min(90 - angle__0, angle__0 - 0) <= min(90 - angle_23, angle_23 - 0):
                        x5, y5, x6, y6 = x_, y_, x0, y0
                    else:
                        x5, y5, x6, y6 = x2, y2, x3, y3
                    d__, d_3, d__3 = abs(A_ * x1 + B_ * y1 + C_) / math.sqrt(A_ * A_ + B_ * B_), abs(
                        A2 * x1 + B2 * y1 + C2) / math.sqrt(A2 * A2 + B2 * B2), abs(A2 * x0 + B2 * y0 + C2) / math.sqrt(
                        A2 * A2 + B2 * B2)
                    A_1, B_1, C_1 = two_points_vline(x5, y5, x6, y6, x1,
                                                     y1) if d__ <= d__3 and d_3 <= d__3 else two_points_line(x5, y5, x6,
                                                                                                             y6, x1, y1)
                elif dis_01 < dis_12:
                    sin, cos = (y2 - y1) / dis_12, (x2 - x1) / dis_12
                    c_x, c_y = x1 + dis_01 * cos, y1 + dis_01 * sin
                    vec_c = (c_x - x0, c_y - y0)
                elif dis_01 > dis_12:
                    sin, cos = (y0 - y1) / dis_01, (x0 - x1) / dis_01
                    c_x, c_y = x1 + dis_12 * cos, y1 + dis_12 * sin
                    vec_c = (x2 - c_x, y2 - c_y)
                else:
                    vec_c = (x2 - x0, y2 - y0)
                if vec_c:
                    angle_c = cal_angle(vec_c, vec_se)
                    A_1, B_1, C_1 = two_points_line(startx, starty, endx, endy, x1,
                                                    y1) if angle_c <= 90 - angle_c else two_points_vline(startx, starty,
                                                                                                         endx, endy, x1,
                                                                                                         y1)
                x_0, y_0 = Drop_point(A_1, B_1, C_1, x0, y0)
                x_2, y_2 = Drop_point(A_1, B_1, C_1, x2, y2)
                area_x_0, area_x_2 = polygon_area([[x1, y1], [x_0, y_0], [x0, y0]]), polygon_area(
                    [[x1, y1], [x_2, y_2], [x2, y2]])
                if 15 <= angle_01 <= 75 and (angle < 60 or 60 <= angle and not h0) and (
                        5 <= angle_0x <= 85 or min12 < 15 or dis__0 >= 2.5 * dis_01):
                    if area_judg(area_x_0, contour_area, 2, seg_type) if angle > 45 else area_judg(area_x_0,
                                                                                                   contour_area, 6,
                                                                                                   seg_type):
                        if rayCasting([x1, y1], appr) != rayCasting([x0, y0], app_0) and cal_ang((x_0, y_0), (x0, y0),
                                                                                                 (x_, y_)) <= 45:
                            pass
                        else:
                            approx.insert(i + 1, [x_0, y_0])
                            conti += 1
                            if step_show: show_result(appro, approx, x1, y1, True)
                    elif area_judg(area_x_0, contour_area, 4, seg_type):
                        c_x_01, c_y_01 = x0 + (x1 - x0) / 3, y0 + (y1 - y0) / 3
                        x_0_u, y_0_u = Drop_point(A_1, B_1, C_1, c_x_01, c_y_01)
                        if rayCasting([x1, y1], appr) != rayCasting([x0, y0], app_0) and cal_ang((x_0_u, y_0_u),
                                                                                                 (x0, y0),
                                                                                                 (x_, y_)) <= 45:
                            pass
                        else:
                            approx.insert(i + 1, [x_0_u, y_0_u])
                            conti += 1
                            if step_show: show_result(appro, approx, x1, y1, True)
                if 15 <= angle_12 <= 75 and (angle < 60 or 60 <= angle and not h1) and (
                        5 <= angle_1x <= 85 or min01 < 15 or dis_23 >= 2.5 * dis_12):
                    if area_judg(area_x_2, contour_area, 2, seg_type) if angle > 45 else area_judg(area_x_2,
                                                                                                   contour_area, 6,
                                                                                                   seg_type):
                        if rayCasting([x1, y1], appr) != rayCasting([x2, y2], app_2) and cal_ang((x_2, y_2), (x2, y2),
                                                                                                 (x3, y3)) <= 45:
                            pass
                        else:
                            conti += 1
                            approx.insert(i, [x_2, y_2])
                            if step_show: show_result(appro, approx, x1, y1, True)
                    elif area_judg(area_x_2, contour_area, 4, seg_type):
                        c_x_12, c_y_12 = x2 + (x1 - x2) / 3, y2 + (y1 - y2) / 3
                        x_2_u, y_2_u = Drop_point(A_1, B_1, C_1, c_x_12, c_y_12)
                        if rayCasting([x1, y1], appr) != rayCasting([x2, y2], app_2) and cal_ang((x_2_u, y_2_u),
                                                                                                 (x2, y2),
                                                                                                 (x3, y3)) <= 45:
                            pass
                        else:
                            conti += 1
                            approx.insert(i, [x_2_u, y_2_u])
                            if step_show: show_result(appro, approx, x1, y1, True)
                if conti == 2:
                    approx.pop(i + 1)
        ##############################################################################################根据结构进行规整
        approx = fill(approx, contour_area, startx, starty, endx, endy, vec_se, seg_type)
        total = len(approx)
        if seg_type == 'bfp' and total > 3:
            no_feature = []
            i = total - 1
            while i > -1:
                x1, y1 = approx[i]
                x2, y2 = approx[i - 1]
                vec_12 = (x2 - x1, y2 - y1)
                angle_12 = cal_angle(vec_12, vec_se);
                min12 = min(angle_12, 90 - angle_12)
                dis_12 = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                no_feature.append([x1, y1])
                no_feature.append([x2, y2])
                end = i - 3 if total == 4 else i - 4 if (total == 5 or total == 6) else i - 5
                for j in range(i - 1, end, -1):
                    if abs(j) > len(approx) or abs(j - 1) > len(approx):
                        i -= 1
                        break
                    x3, y3 = approx[j]
                    x4, y4 = approx[j - 1]
                    vec_34 = (x4 - x3, y4 - y3)
                    dis_34 = math.sqrt((y4 - y3) ** 2 + (x4 - x3) ** 2)
                    angle_34 = cal_angle(vec_34, vec_se);
                    min34 = min(angle_34, 90 - angle_34)
                    angle_13 = cal_angle(vec_12, vec_34, half=False)
                    if [x3, y3] != no_feature[-1]:
                        no_feature.append([x3, y3])
                    no_feature.append([x4, y4])
                    if (angle_13 <= 10 or angle_13 >= 170 or 80 <= angle_13 <= 100) and (
                            min12 <= 10 and min34 <= 15 or min34 <= 10 and min12 <= 15) and (
                            dis_12 >= 6 and dis_34 >= 6):
                        if j == i - 1:
                            i = j
                            break
                        no_feature, show, x, y = del_no_feature_process(no_feature, angle_13, angle_12, angle_34, minx,
                                                                        miny, contour_area, vec_se, startx, starty,
                                                                        endx, endy, approx, l1, seg_type)
                        after = len(no_feature)
                        if after <= i - j + 2:
                            n = 0
                            for m in range(i, j - 2, -1):
                                if n < after:
                                    approx[m] = no_feature[n]
                                    n += 1
                                else:
                                    approx[m] = [-1, -1]
                            for m in range(len(approx) - 1, -1, -1):
                                if approx[m] == [-1, -1]:
                                    approx.pop(m)
                        else:
                            n = 0
                            for m in range(i, j - 2, -1):
                                approx[m] = no_feature[n]
                                n += 1
                            if j - 1 < 0:
                                l = -1
                                for m in range(after - i + j - 2):
                                    approx.insert(j - 1, no_feature[l])
                                    l -= 1
                            else:
                                for m in range(after - i + j - 2):
                                    approx.insert(j - 1, no_feature[n])
                                    n += 1
                        i = j
                        if step_show: show_result(appro, approx, x1, y1, True)
                        break
                    elif j == end + 1:
                        i -= 1
                no_feature = []
            approx, show = spin(approx, contour, contour_area, startx, starty, endx, endy, vec_se, seg_type, False)
        elif seg_type != 'bfp' and total > 3:
            approx, show = spin(approx, contour, contour_area, startx, starty, endx, endy, vec_se, seg_type, False)
        elif total == 3:
            approx, _ = zhu_rect(startx, starty, endx, endy, vec_se, contour, box, box_area)
        if hierarchy[
            0, c, 3] == -1: ostartx, ostarty, oendx, oendy, ovec_se, index = startx, starty, endx, endy, vec_se, c
        approx = np.array(approx)
        # im=cv2.drawContours(im,[approx],-1, 255, 1)
        final.append(approx)
        inter_outer.append([c, hierarchy[0, c, 3]])
    # io.imsave('test_.tif',im)
    return final, inter_outer


VECTOR_DRIVER = {
    "shp": "ESRI Shapefile",
    "json": "GeoJSON",
    "geojson": "GeoJSON"
}


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


def ImageRowCol2Projection(adfGeoTransform, iCol, iRow):
    dProjX = adfGeoTransform[0] + adfGeoTransform[1] * iCol + adfGeoTransform[2] * iRow
    dProjY = adfGeoTransform[3] + adfGeoTransform[4] * iCol + adfGeoTransform[5] * iRow
    return (dProjX, dProjY)


def merge(in_data, strVectorFile, seg_type, pix, xoff=None, yoff=None, xsize=None, ysize=None):
    map_dict = {"resident": {"ClsCode": 255, "ClsName": "resident", "LabelCode": "DW0603", "LabelName": "居民区"},
                "bfp": {"ClsCode": 255, "ClsName": "bfp", "LabelCode": "DW0601", "LabelName": "稀疏房屋"},
                "water": {"ClsCode": 255, "ClsName": "water", "LabelCode": "DW08", "LabelName": "水体"},
                "road": {"ClsCode": 255, "ClsName": "road", "LabelCode": "DW07", "LabelName": "道路"},
                "vegetation": {"ClsCode": 255, "ClsName": "vegetation", "LabelCode": "DW11", "LabelName": "植被"},
                "cloud": {"ClsCode": 255, "ClsName": "cloud", "LabelCode": "DW12", "LabelName": "云"}}

    dataset = gdal.Open(in_data)
    xoff = 0 if xoff is None else xoff
    yoff = 0 if yoff is None else yoff
    xsize = dataset.RasterXSize if xsize is None else xsize
    ysize = dataset.RasterYSize if ysize is None else ysize
    w = xsize
    h = ysize

    # outbandsize = dataset.RasterCount      #获取通道数
    im_geotrans = dataset.GetGeoTransform()  # 获取地理坐标信息  GeoTransform[0],GeoTransform[3]左上角位置，GeoTransform[1]是像元宽度，GeoTransform[5]是像元高度
    im_geotrans = list(im_geotrans)
    im_geotrans[0] += xoff * im_geotrans[1]
    im_geotrans[3] += yoff * im_geotrans[5]

    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    ogr.RegisterAll()  # 注册所有格式文件驱动

    strDriverName = VECTOR_DRIVER.get(os.path.splitext(strVectorFile)[1][1:].lower())
    oDriver = ogr.GetDriverByName(strDriverName)  # 得到shape处理器
    oDS = oDriver.CreateDataSource(strVectorFile)  # 创建shp文件

    srs = osr.SpatialReference()
    srs.ImportFromWkt(dataset.GetProjectionRef())
    oLayer = oDS.CreateLayer("TestPolygon", srs, ogr.wkbPolygon, options=['ENCODING=UTF-8'])  # 创建图层
    if oLayer is None:
        print("oLayer is None\n")
    oLayer.CreateField(ogr.FieldDefn("ClsValue", ogr.OFTInteger))
    oLayer.CreateField(ogr.FieldDefn("ClsCode", ogr.OFTInteger))
    oLayer.CreateField(ogr.FieldDefn("ClsName", ogr.OFTString))
    oLayer.CreateField(ogr.FieldDefn("LabelCode", ogr.OFTString))
    oLayer.CreateField(ogr.FieldDefn("LabelName", ogr.OFTString))
    oDefn = oLayer.GetLayerDefn()  # 获取属性表头信息

    c = 0

    img = dataset.ReadAsArray(xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize)
    contours, inter_outer = regular(img, seg_type, pix)
    assert len(contours) == len(inter_outer)
    while c < len(contours):
        contour, index, ioro = contours[c], inter_outer[c][0], inter_outer[c][1]
        if ioro == -1:
            garden1 = ogr.Geometry(ogr.wkbPolygon)
            oFeatureTriangle = ogr.Feature(oDefn)
            oFeatureTriangle.SetField(0, map_dict.get(seg_type).get('ClsCode'))
            oFeatureTriangle.SetField(1, map_dict.get(seg_type).get('ClsCode'))
            oFeatureTriangle.SetField(2, map_dict.get(seg_type).get('ClsName'))
            oFeatureTriangle.SetField(3, map_dict.get(seg_type).get('LabelCode'))
            oFeatureTriangle.SetField(4, map_dict.get(seg_type).get('LabelName'))
            box1 = ogr.Geometry(ogr.wkbLinearRing)
            for point in contour:
                x_col = float(point[1])
                y_row = float(point[0])
                coordinate_x, coordinate_y = ImageRowCol2Projection(im_geotrans, y_row, x_col)  # 行、列转成地理坐标信息
                box1.AddPoint(coordinate_x, coordinate_y)
            box1.CloseRings()
            garden1.AddGeometry(box1)
            if c == len(contours) - 1:
                c += 1
            for j in range(c + 1, len(contours)):
                contour_, index_, ioro_ = contours[j], inter_outer[j][0], inter_outer[j][1]
                if ioro_ != -1 and index_ == index + 1:
                    inner = ogr.Geometry(ogr.wkbLinearRing)
                    for point in contour_:
                        x_col = float(point[1])
                        y_row = float(point[0])
                        coordinate_x, coordinate_y = ImageRowCol2Projection(im_geotrans, y_row, x_col)
                        inner.AddPoint(coordinate_x, coordinate_y)
                    inner.CloseRings()
                    garden1.AddGeometry(inner)
                    index += 1
                    c += 1
                else:
                    c += 1
                    break
            geomTriangle = ogr.CreateGeometryFromWkt(str(garden1))
            oFeatureTriangle.SetGeometry(geomTriangle)
            oLayer.CreateFeature(oFeatureTriangle)
        else:
            c += 1
    oDS.Destroy()
    dataset = None


def process_boundary(raster_path, shp_path, seg_type='bfp', resolution=0.5, xoff=None, yoff=None, xsize=None, ysize=None):
    pix = resolution
    merge(raster_path, shp_path, seg_type, pix, xoff, yoff, xsize, ysize)