"""
Usage:
  python generate_tfrecord.py \
    --csv_input=<path_to_csv_file> \
    --output_path=<filename.record> \
    --image_dir=<path_to_images> \
    --label_map=<path_to_labelmap_file>
"""
from __future__ import absolute_import, division, print_function

import io
import os
import re
from collections import namedtuple

from PIL import Image

import utils
import pandas as pd
import tensorflow as tf

flags = tf.compat.v1.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
flags.DEFINE_string('label_map', '', 'Path to labelmap')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
# def class_text_to_int(row_label):
#     if row_label == 'ball':
#         return 1
#     elif row_label == 'goal post':
#         return 2
#     else:
#         None

def class_text_to_int(labelmap_file, row_label):
    try:
        with open(labelmap_file, 'r') as foo:
            text = foo.read()
            categories = re.findall(r'name: \'(.*)\'', text)
        return (categories.index(row_label) + 1)
    except Exception:
        return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename,
            x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)),
                           'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(FLAGS.label_map, row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': utils.int64_feature(height),
        'image/width': utils.int64_feature(width),
        'image/filename': utils.bytes_feature(filename),
        'image/source_id': utils.bytes_feature(filename),
        'image/encoded': utils.bytes_feature(encoded_jpg),
        'image/format': utils.bytes_feature(image_format),
        'image/object/bbox/xmin': utils.float_list_feature(xmins),
        'image/object/bbox/xmax': utils.float_list_feature(xmaxs),
        'image/object/bbox/ymin': utils.float_list_feature(ymins),
        'image/object/bbox/ymax': utils.float_list_feature(ymaxs),
        'image/object/class/text': utils.bytes_list_feature(classes_text),
        'image/object/class/label': utils.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.compat.v1.app.run()
