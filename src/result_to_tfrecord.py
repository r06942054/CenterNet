import tensorflow as tf
import json
import numpy as np
import argparse
import sys

sys.path.append("/home/omnieyes/renjie/TF_models/research/")
from object_detection.core.standard_fields import TfExampleFields
import sys
sys.path.append("/home/omnieyes/renjie/OmniEyes_forOD/OmniEyes/")
from data_post_processing.util import util


parser = argparse.ArgumentParser(description='Labeled video(s) to images, grouped by folder with names as video names.')
parser.add_argument('-gt', '--gt-json-path', help='Path to GT json file.')
parser.add_argument('-rt', '--result-json-path', help='Path to result json file.')
parser.add_argument('-o', '--output-path', help='Path to output tfrecord file.')

gt_json_path = args.gt_json_path
result_json_path = args.result_json_path
output_path = args.output_path


with open(gt_json_path, 'r') as f:
    gt_json = json.load(f)
with open(result_json_path, 'r') as f:
    result_list = json.load(f)

categories_dict = {}
for d in gt_json['categories']:
    categories_dict[d['id']] = d['name']


record_dict = {}
for d in gt_json['images']:
    image_dict = {}
    image_dict['filename'] = d['file_name']
    image_dict['width'] = d['width']
    image_dict['height'] = d['height']
    image_dict['detection_box'] = []
    image_dict['detection_score'] = []
    image_dict['detection_label'] = []
    image_dict['object_label'] = []
    image_dict['object_text'] = []
    image_dict['object_box'] = []
    record_dict[d['id']] = image_dict


for d in gt_json['annotations']:
    record_dict[d['image_id']]['object_label'].append(d['category_id'])
    record_dict[d['image_id']]['object_text'].append(categories_dict[d['category_id']].encode('utf-8'))

    bbox = d['bbox']
    xmin = bbox[0] / record_dict[d['image_id']]['width']
    ymin = bbox[1] / record_dict[d['image_id']]['height']
    xmax = (bbox[0] + bbox[2]) / record_dict[d['image_id']]['width']
    ymax = (bbox[1] + bbox[3]) / record_dict[d['image_id']]['height']
    record_dict[d['image_id']]['object_box'].append([xmin, ymin, xmax, ymax])


for d in result_list:
    record_dict[d['image_id']]['detection_score'].append(d['score'])
    record_dict[d['image_id']]['detection_label'].append(d['category_id'])

    bbox = d['bbox']
    xmin = bbox[0] / record_dict[d['image_id']]['width']
    ymin = bbox[1] / record_dict[d['image_id']]['height']
    xmax = (bbox[0] + bbox[2]) / record_dict[d['image_id']]['width']
    ymax = (bbox[1] + bbox[3]) / record_dict[d['image_id']]['height']
    
    record_dict[d['image_id']]['detection_box'].append([xmin, ymin, xmax, ymax])


for key in record_dict.keys():
    sort_index = np.argsort(record_dict[key]['detection_score'])[::-1]
    record_dict[key]['detection_score'] = np.array(record_dict[key]['detection_score'])[sort_index]
    record_dict[key]['detection_label'] = np.array(record_dict[key]['detection_label'])[sort_index]
    record_dict[key]['detection_box'] = np.array(record_dict[key]['detection_box'])[sort_index]
    record_dict[key]['object_box'] = np.array(record_dict[key]['object_box'])

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# writer TODO: better naming
writer = tf.python_io.TFRecordWriter(output_path)
key = 1
empty_count = 0
for key in record_dict.keys():
    # print(record_dict[key]['detection_box'])
    if len(record_dict[key]['detection_box'])==0: 
        empty_count += 1
        continue
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        TfExampleFields.height: _int64_feature(record_dict[key]['height']),
        TfExampleFields.width: _int64_feature(record_dict[key]['width']),
        TfExampleFields.filename: _bytes_feature(record_dict[key]['filename'].encode('utf-8')),
        TfExampleFields.source_id: _bytes_feature(record_dict[key]['filename'].encode('utf-8')),
        TfExampleFields.image_format: _bytes_feature('jpg'.encode('utf-8')),
        TfExampleFields.object_bbox_xmin: _float_list_feature(record_dict[key]['object_box'][: ,0]),
        TfExampleFields.object_bbox_xmax: _float_list_feature(record_dict[key]['object_box'][: ,2]),
        TfExampleFields.object_bbox_ymin: _float_list_feature(record_dict[key]['object_box'][: ,1]),
        TfExampleFields.object_bbox_ymax: _float_list_feature(record_dict[key]['object_box'][: ,3]),
        TfExampleFields.object_class_text: _bytes_list_feature(record_dict[key]['object_text']),
        TfExampleFields.object_class_label: _int64_list_feature(record_dict[key]['object_label']),
        TfExampleFields.detection_bbox_xmin: _float_list_feature(record_dict[key]['detection_box'][: ,0]),
        TfExampleFields.detection_bbox_xmax: _float_list_feature(record_dict[key]['detection_box'][: ,2]),
        TfExampleFields.detection_bbox_ymin: _float_list_feature(record_dict[key]['detection_box'][: ,1]),
        TfExampleFields.detection_bbox_ymax: _float_list_feature(record_dict[key]['detection_box'][: ,3]),
        TfExampleFields.detection_class_label: _int64_list_feature(record_dict[key]['detection_label']),
        TfExampleFields.detection_score: _float_list_feature(record_dict[key]['detection_score']),
    }))

    writer.write(tf_example.SerializeToString())

writer.close()