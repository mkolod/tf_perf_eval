from time import time
import multiprocessing 
import numpy as np
import tensorflow as tf

num_repeat = 100000
file_list = ['/tf_jpeg_perf/ElCapitan_256_by_256.jpg'] * num_repeat

num_inter_threads = multiprocessing.cpu_count() / 2
num_intra_threads = multiprocessing.cpu_count() / 2

print("Running multi-threaded JPEG decode in TensorFlow. Please wait.\n")


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

batch_size = 1000

filename_queue = tf.train.string_input_producer(string_tensor = file_list,
                                                    shuffle = False,
                                                    seed = 42,
                                                    capacity = num_intra_threads * batch_size,
                                                    shared_name = 'filename_queue',
                                                    name = 'filename_queue',
                                                    num_epochs = 1)

sess.run(tf.initialize_local_variables())

reader_coord = tf.train.Coordinator()

reader = tf.WholeFileReader()

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
    
    encoded_image = ex_dequeue_op
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
    pass
finally:
    reader_coord.request_stop()
    process_coord.request_stop()

reader_coord.join(internal_tf_threads)
process_coord.join(process_threads)
sess.close()


end = time()
total_time = end - start
im_sec = float(count) / total_time

print("Running time: %.2f seconds" % total_time)
print("Read %d images" % count)
print("Images/second: %.2f" % im_sec)
print("Images/second/core: %.2f" % (im_sec / num_inter_threads))