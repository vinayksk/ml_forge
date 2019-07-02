import tensorflow as tf
import csv 
import io
from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('training_output_path', '', 'Path to output tfrecords')
flags.DEFINE_string('testing_output_path', '', 'Path to output tfrecords for testing')
FLAGS = flags.FLAGS

def convert_to_list(string, cat):
    temp = string[1:-1]
    if cat == "int":
        arr = [int(i) for i in temp.split(", ")]
    elif cat == "float":
        arr = [float(i) for i in temp.split(", ")]
    else:
        arr = temp.split(", ")
    return arr
 
def create_tf_example(example):
  # Populate the following variables from your example.
  height = long(480) # Image height
  width = long(640) # Image width
  filename = example['name'].encode('utf') # Filename of the image. Empty if image is not from file
  with tf.gfile.GFile(example['name'], 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_image_data = encoded_jpg # Encoded image bytes
  image_format = b'jpg' # b'jpeg' or b'png'

  xmins = [x/width for x in convert_to_list(example['xmins'], "float")] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [x/width for x in convert_to_list(example['xmaxs'], "float")] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [y/height for y in convert_to_list(example['ymins'], "float")] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [y/height for y in convert_to_list(example['ymaxs'], "float")] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [ct.encode('utf') for ct in convert_to_list(example['classes_text'], "str")] # List of string class name of bounding box (1 per box)
  classes = convert_to_list(example['classes'], "int") # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  train_writer = tf.python_io.TFRecordWriter(FLAGS.training_output_path)
  test_writer = tf.python_io.TFRecordWriter(FLAGS.testing_output_path)

  # Write code to read in your dataset to examples variable
  examples = csv.DictReader(open("annotations.csv"))
  
  count = 0 
  for example in examples:
    tf_example = create_tf_example(example)
    if count > 159:
        test_writer.write(tf_example.SerializeToString())
    else:
        train_writer.write(tf_example.SerializeToString())
    count = count + 1
  train_writer.close()
  test_writer.close()


if __name__ == '__main__':
  tf.app.run()
