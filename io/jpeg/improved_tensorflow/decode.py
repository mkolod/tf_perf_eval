import tensorflow as tf
from tensorflow.python.platform import flags
from time import time

flags.DEFINE_integer('max_steps', 100, 'max number of steps to run')
flags.DEFINE_integer('warmup', 10, 'max number of steps to warmup')
flags.DEFINE_integer('batch_size', 1000, 'batch size')

FLAGS = flags.FLAGS

def main(_):
  image_name = '../tensorflow/ElCapitan_256_by_256.jpg'
  file_list = [image_name]

  print("Running multi-threaded JPEG decode in TensorFlow. Please wait.\n")

  sess = tf.Session('', config=tf.ConfigProto())

  batch_size = FLAGS.batch_size

  filename_queue = tf.FIFOQueue(-1, tf.string)
  filename_enq_ops = []
  for i in xrange(batch_size):
    filename_enq_op = filename_queue.enqueue(image_name)
    filename_enq_ops.append(filename_enq_op)
  filename_enq_loop = tf.group(*filename_enq_ops)

  image_queue = tf.FIFOQueue(-1, tf.uint8)
  enq_ops = []
  for i in xrange(batch_size):
    with tf.control_dependencies([tf.constant(i)]):
      reader = tf.WholeFileReader()
      _, encoded_image = reader.read(filename_queue)
      decoded = tf.image.decode_jpeg(encoded_image, channels = 3)
    enq_op = image_queue.enqueue(decoded)
    enq_ops.append(enq_op)
  enq_loop = tf.group(*enq_ops)

  deq_ops = []
  for i in xrange(batch_size):
    deq_op = image_queue.dequeue()
  deq_ops.append(deq_op)
  deq_loop = tf.group(*deq_ops)

  sess.run(tf.initialize_local_variables())
  tf.train.start_queue_runners(sess)

  count = 0
  i = 0;
  for i in xrange(FLAGS.max_steps + FLAGS.warmup):
    if i == FLAGS.warmup:
      count = 0
      start = time()
    print("step_count: %d" % (i))
    sess.run([filename_enq_loop, enq_loop, deq_loop])
    count += batch_size

  end = time()
  total_time = end - start
  im_sec = float(count) / total_time

  print("Running time: %.2f seconds" % total_time)
  print("Read %d images" % count)
  print("Images/second: %.2f" % im_sec)


if __name__ == '__main__':
  tf.app.run()

