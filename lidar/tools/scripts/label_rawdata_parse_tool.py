import json
import os
import sys
import argparse
import traceback


def parse_type(label_string):
    if label_string == "其他（可移动）":
        return 0
    elif label_string == "二轮车":
        return 1
    elif label_string == "车辆":
        return 2
    elif label_string == "其他（静止）":
        return 3
    elif label_string == "行人":
        return 4
    else:
        return 0


def parse_baidu_raw_label_file(file_path, out_put_path):
    with open(file_path, 'r') as f_in:
        for line in f_in:
            try:
                strs = line.split("\t")
                if strs[0] == "local_path":
                    continue
                json_str = strs[2].strip()
                json_obj = json.loads(json_str)
                url = json_obj['url']
                zip_file_name = url.split("/")[-1]
                file_index = zip_file_name.split(".")[0]
                pcd_index = file_index.split("_")[-1]
                extra = json_obj['extra']
                out_json_file = os.path.join(out_put_path, pcd_index + ".label")
                with open(out_json_file, 'w') as f_out:
                    for item in extra:
                        labels = item['label']['3D']       
                        for label in labels:
                            center_x = float(label['position']['x'])
                            center_y = float(label['position']['y'])
                            center_z = float(label['position']['z'])
                            length = float(label['size'][0])
                            width = float(label['size'][1])
                            height = float(label['size'][2])
                            phi = float(label['rotation']['phi'])
                            cls_type = parse_type(label['type'])
                            f_out.write("%f %f %f %f %f %f %f %s\n" % (
                                center_x, center_y, center_z,
                                length, width, height, phi, cls_type
                            ))
            except Exception as e:
                print('illegal str {}'.format(line))
                traceback.print_exc()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='args for baidu raw data parser')
    parser.add_argument('--input_path', '-i',
                        help='input path of raw label file', required=True)
    parser.add_argument('--output_path', '-o',
                        help='output path of label json result', required=True)
    args = parser.parse_args()
    parse_baidu_raw_label_file(args.input_path, args.output_path)
