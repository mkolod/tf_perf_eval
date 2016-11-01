FROM gcr.io/tensorflow/tensorflow:0.11.0rc2
RUN mkdir /tf_jpeg_perf
COPY ElCapitan_256_by_256.jpg /tf_jpeg_perf
COPY multithreaded_jpeg_decode.py /tf_jpeg_perf

CMD /bin/bash -c "cd /tf_jpeg_perf && python multithreaded_jpeg_decode.py"
