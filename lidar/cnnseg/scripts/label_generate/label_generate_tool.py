#!/usr/bin/env python

import os
import math
import argparse
class LabelPoint(object):
    def __init__(self, x, y, z, pos_x=-1, pos_y=-1):
        # 3D pos
        self._x = x
        self._y = y
        self._z = z
        # 2D gird pos
        self._pos_x = pos_x
        self._pos_y = pos_y

class LabelObject(object):
    def __init__(self, length, width, height, center, phi, cls_type, point_list):
        self._length = length
        self._width = width
        self._height = height
        self._center = center
        self._phi = phi
        self._cls_type = cls_type
        self._point_list = point_list

    def __str__(self):
        return 'length:%f | width:%f | height:%f | center_x:%f | center_y:%f | center_z:%f | phi:%f | type:%s' % \
            (self._length, self._width, self._height, self._center._x,
             self._center._y, self._center._z, self._phi, self._cls_type)


class LabelGenerator(object):
    def __init__(self, label_raw_input_path, label_grid_output_path, width, height, lidar_range):
        self._label_raw_input_path = label_raw_input_path
        self._label_grid_output_path = label_grid_output_path
        self._width = width
        self._height = height
        self._lidar_range = lidar_range

    def filter_point(self, x, y, polygon_point_list, count_thr):
        if len(polygon_point_list) != count_thr:
            return False
        n_cross = 0
        for i in range(0, count_thr):
            p1 = polygon_point_list[i]
            p2 = polygon_point_list[(i+1) % count_thr]
            if p1._pos_y == p2._pos_y:
                continue
            if y < min(p1._pos_y, p2._pos_y) or y >= max(p1._pos_y, p2._pos_y):
                continue
            x_flag = (float)(y - p1._pos_y) * (float)(p2._pos_x - p1._pos_x) / (float)(p2._pos_y - p1._pos_y) + p1._pos_x
            if x_flag > x:
                n_cross += 1
        if n_cross % 2 == 1:
            return True
        else:
            return False

    def generate_label(self):
        for label_file in os.listdir(self._label_raw_input_path):
            label_obj_list = []
            print("parsing label " + label_file)
            with open(os.path.join(self._label_raw_input_path, label_file), 'r') as f_read:
                lines = f_read.readlines()
                print("get " + str(len(lines)) + " object")
                for line in lines:
                    parts = line.strip("\r\n").split("\t")
                    if len(parts) == 1:
                        parts = line.strip("\r\n").split(" ")
                    center_x = float(parts[0])
                    center_y = float(parts[1])
                    center_z = float(parts[2])
                    length = float(parts[3])
                    width = float(parts[4])
                    height = float(parts[5])
                    phi = float(parts[6])
                    cls_type = int(parts[7])
                    center = LabelPoint(x=center_x, y=center_y, z=center_z)
                    label_obj = LabelObject(length=length, width=width, height=height,
                                            center=center, phi=phi, cls_type=cls_type, point_list=[])
                    label_obj_list.append(label_obj)
            label_result = self.generate_2d_label(label_obj_list)
            print("label_result contains {:d} objects".format(len(label_result)))
            self.output_result(label_file.split('.')[0], label_result)

    def generate_2d_label(self, label_list):
        inv_res_x = 0.5 * float(self._width)/self._lidar_range
        meter_per_pixel = self._lidar_range * 2.0 / float(self._width)
        label_result = []
        for i in range(0, len(label_list)):
            label_object = label_list[i]
            phi = label_object._phi
            center_x = label_object._center._x
            center_y = label_object._center._y
            center_pos_x ,center_pos_y = self.group_pc_2pixel(center_x,center_y,inv_res_x)
            length = label_object._length
            width = label_object._width
            min_x = self._width
            min_y = self._height
            max_x = -1
            max_y = -1
            # get four point
            right_top = LabelPoint(center_x + length / 2.0, center_y + width / 2.0, 0, 0, 0)
            right_bottom = LabelPoint(center_x + length / 2.0, center_y - width / 2.0, 0, 0, 0)
            left_top = LabelPoint(center_x - length / 2.0,center_y + width / 2.0, 0, 0, 0)
            left_bottom = LabelPoint(center_x - length / 2.0, center_y - width / 2.0, 0, 0, 0)
            # rotate point by phi ,tpush_backhe push back sequence is important ,must be Clockwise from right top
            label_object._point_list.append(self.get_rotate_point(right_top, phi, label_object._center))
            label_object._point_list.append(self.get_rotate_point(right_bottom, phi, label_object._center))
            label_object._point_list.append(self.get_rotate_point(left_bottom, phi, label_object._center))
            label_object._point_list.append(self.get_rotate_point(left_top, phi, label_object._center))
            # get 2d points belong to the 2d bbox
            for label_point in label_object._point_list:
                if label_point._pos_x < min_x:
                    min_x = label_point._pos_x
                if label_point._pos_y < min_y:
                    min_y = label_point._pos_y
                if label_point._pos_x > max_x:
                    max_x = label_point._pos_x
                if label_point._pos_y > max_y:
                    max_y = label_point._pos_y
            for row in range(min_y, max_y+1):
                for col in range(min_x, max_x+1):
                    if self.filter_point(col, row, label_object._point_list, 4):
                        tmp_result = []
                        # category score list
                        category_scores = [0.0] * 5
                        category_scores[label_object._cls_type] = 1.0
                        # cal instance label
                        instance_x, instance_y = self.generate_label_instance(center_pos_y - row, center_pos_x - col)
                        instance_x *= meter_per_pixel
                        instance_y *= meter_per_pixel
                        # cal height label
                        height = label_object._height
                        # cal heading label
                        heading_x = math.cos(phi)
                        heading_y = math.sin(phi)
                        # cal confidence label
                        confidence_score = 1.0
                        category_score = 1.0
                        tmp_result.extend([row, col, instance_x, instance_y, category_score, confidence_score])
                        tmp_result.extend(category_scores)
                        tmp_result.extend([heading_x, heading_y, height])
                        tmp_result = [str(x) for x in tmp_result]
                        label_result.append(" ".join(tmp_result) + "\n")
        return label_result

    def generate_label_instance(self, row_offset, col_offset):
        # scale = max(abs(row_offset, col_offset))
        # return row_offset / scale, col_offset / scale
        return row_offset, col_offset

    def get_rotate_point(self, point, rz, origin_point):
        inv_res_x = 0.5 * float(self._width) / self._lidar_range
        x = (point._x - origin_point._x)*math.cos(rz) - (point._y - origin_point._y)*math.sin(rz) + origin_point._x
        y = (point._x - origin_point._x)*math.sin(rz) + (point._y - origin_point._y)*math.cos(rz) + origin_point._y
        # map point cloud point to 2d ,for axis rotated case
        pos_x, pos_y = self.group_pc_2pixel(x, y, inv_res_x)
        return LabelPoint(x, y, 0, pos_x, pos_y)

    def group_pc_2pixel(self, pc_x, pc_y, scale):
        fx = (self._lidar_range - (0.707107 * (pc_x + pc_y))) * scale
        fy = (self._lidar_range - (0.707107 * (pc_x - pc_y))) * scale
        x = -1 if(fx < 0) else int(fx)
        y = -1 if(fy < 0) else int(fy)
        return x, y

    def output_result(self, file_prefix, result_list):
        file_name = file_prefix + ".txt"
        full_path = os.path.join(self._label_grid_output_path, file_name)
        with open(full_path, 'w') as f_w:
          f_w.writelines(result_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args for label generate')
    parser.add_argument('--input_path', '-i', help='input path of json label file', required=True)
    parser.add_argument('--output_path', '-o', help='output path of label result', required=True)
    parser.add_argument('--width', '-wh', help='2d width', type=int, default=480)
    parser.add_argument('--height', '-ht', help='2d height', type=int, default=480)
    parser.add_argument('--range', '-r', help='lidar range', type=float, default=40.0)
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    generator = LabelGenerator(args.input_path, args.output_path, args.width, args.height, args.range)
    generator.generate_label()
