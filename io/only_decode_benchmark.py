from datetime import datetime
import multiprocessing
import os
from os import listdir
from os.path import isfile, join
from time import time

import numpy as np

import tensorflow as tf

proto_in = '/datasets/resized_imagenet/training/'
# proto_in = '/datasets/imagenet_original'
only_files = [f for f in listdir(proto_in) if isfile(join(proto_in, f))]

# make sure we have 1,024 training and 128 validation proto files
only_train_files = [os.path.join(proto_in, f) for f in only_files if "train" in f]
only_validation_files = [os.path.join(proto_in, f) for f in only_files if "valid" in f]

print("number of training files: %d" % len(only_train_files))
print("number of validation files: %d" % len(only_validation_files))

num_inter_threads = multiprocessing.cpu_count() / 2
num_intra_threads = multiprocessing.cpu_count() / 2

feature_map = {
'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                    default_value=''),
'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                        default_value=-1),
'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                       default_value='')
}
sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
feature_map.update(
    {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                 'image/object/bbox/ymin',
                                 'image/object/bbox/xmax',
                                 'image/object/bbox/ymax']})

sess = tf.Session(config=tf.ConfigProto(
    inter_op_parallelism_threads=num_inter_threads,
    intra_op_parallelism_threads=num_intra_threads))

batch_size = 1024

data_files = only_train_files[0:100]

filename_queue = tf.train.string_input_producer(string_tensor = data_files,
                                                    shuffle = False,
                                                    seed = 42,
                                                    capacity = num_intra_threads * batch_size,
                                                    shared_name = 'filename_queue',
                                                    name = 'filename_queue',
                                                    num_epochs = 1)

sess.run(tf.initialize_local_variables())

reader_coord = tf.train.Coordinator()

reader = tf.TFRecordReader()
_, read_op = reader.read(filename_queue)

examples_queue = tf.FIFOQueue(
    capacity= num_inter_threads * batch_size,
    dtypes=[tf.string])

ex_enqueue_op = examples_queue.enqueue(read_op)

examples_qr = tf.train.QueueRunner(examples_queue, [ex_enqueue_op] * num_inter_threads)
process_coord = tf.train.Coordinator()
process_threads = examples_qr.create_threads(sess, coord=process_coord, start=True)

processed_tensor_tuples = []

for thread in xrange(num_inter_threads):
        
    ex_dequeue_op = examples_queue.dequeue()
    proto_parse_op = tf.parse_single_example(ex_dequeue_op, feature_map)
    
    encoded_image = proto_parse_op['image/encoded']
    decoded = tf.image.decode_jpeg(encoded_image, channels = 3)  
    
    # dummy op to force use of jpeg decoding
    processed_tensor_tuples.append([tf.shape(decoded)])

joined = tf.train.batch_join(
    processed_tensor_tuples, batch_size = batch_size, capacity = 2 * num_inter_threads * batch_size)

internal_tf_threads = tf.train.start_queue_runners(sess=sess, coord=reader_coord)    

count = 0

start = time()

try:
    
    while not (reader_coord.should_stop() or process_coord.should_stop()):
        
        value = sess.run(joined)
        count += len(value)
        
        
except tf.errors.OutOfRangeError:
    print('Done')
finally:
    reader_coord.request_stop()
    process_coord.request_stop()

reader_coord.join(internal_tf_threads)
process_coord.join(process_threads)
sess.close()


end = time()
total_time = end - start

print("Running time: %.2f seconds" % total_time)
print("Images/second: %.2f" % (float(count) / total_time))
