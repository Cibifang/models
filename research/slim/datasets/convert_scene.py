"""Converts Scene Classification data to TFRecords of TF-Example protos.

This module reads the files that make up the Scene Classification data and
creates two TFRecord datasets: one for train and one for test. Each TFRecord
dataset is comprised of a set of TF-Example protocol buffers, each of which
contain a single image and label.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import json

import tensorflow as tf
import dataset_utils

tf.app.flags.DEFINE_string(
    'source_dir', None, 'The directory where the source files are stored.')

tf.app.flags.DEFINE_string(
    'split_name', 'train', 'The name of the train/validation split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'num_per_shard', 500, 'The number of photos in per tfdata file')

FLAGS = tf.app.flags.FLAGS

LABEL_FILE = 'scene_classes.csv'


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(source_dir, split_name):
  """Returns a list of filenames and inferred class names.

  Args:
    source_dir: A directory containing a subdirectory containing JPG images and
      a json file with some information about classes.
    split_name: The name of the dataset, either 'train' or 'validation'.

  Returns:
    A list of image file names with path and corresponding class id.
  """
  scene_root = os.path.join(source_dir, 'scene_photos')
  class_file = os.path.join(source_dir, 'scene_photos.json')
  photos = []

  with open(class_file, 'r') as f:
    data = json.load(f)

    for image in data:
      photo = {}
      photo['filename'] = os.path.join(scene_root, image['image_id'])
      photo['label_id'] = int(image['label_id'])
      photo['image_name'] = image['image_id'].encode()

      photos.append(photo)

  return photos


def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
  output_filename = 'scene_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, num_shards)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, photos, dataset_dir, num_per_shard):
  """Converts the given photos to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    photos: A list of image file paths and corresponding class id.
    dataset_dir: The directory where the converted datasets are stored.
    num_per_shard: The number of photos in per tfdata file
  """
  assert split_name in ['train', 'validation', 'test']

  num_shards = int(math.ceil(len(photos) / float(num_per_shard)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(num_shards):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id, num_shards)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(photos))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(photos), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(photos[i]['filename'], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_id = photos[i]['label_id']

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id,
                photos[i]['image_name'])
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def read_label(label_file):
  """Read the source label file

  Args:
    label_file: the source label file

  Returns:
    A map of (integer) labels to class names.
  """
  labels = {}

  with open(label_file, 'r') as f:
    for line in f.readlines():
      label = int(line.split(",")[0])
      class_name = line.split(",")[2]
      labels[label] = class_name

  return labels


def run(source_dir, dataset_dir, split_name, num_per_shard):
  """Runs the conversion operation.

  Args:
    source_dir: The directory where the dataset needed to convert is stored.
    dataset_dir: The directory where the converted dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  # if _dataset_exists(dataset_dir):
  #   print('Dataset files already exist. Exiting without re-creating them.')
  #   return

  assert split_name in ['train', 'validation']

  photos = _get_filenames_and_classes(source_dir, split_name)

  _convert_dataset(split_name, photos, dataset_dir, num_per_shard)

  label_file = os.path.join(source_dir, LABEL_FILE)
  labels_to_class_names = read_label(label_file)
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  print('\nFinished converting the Scene dataset!')

def main(_):
  if not FLAGS.source_dir:
    raise ValueError(
      'You must supply the source directory with --source_dir')

  if not FLAGS.dataset_dir:
    raise ValueError(
      'You must supply the dataset directory with --dataset_dir')

  print('start')

  run(FLAGS.source_dir, FLAGS.dataset_dir, FLAGS.split_name,
      FLAGS.num_per_shard)

if __name__ == '__main__':
  tf.app.run()
