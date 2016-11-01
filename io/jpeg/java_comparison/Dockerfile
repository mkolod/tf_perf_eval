FROM ubuntu:16.04
RUN mkdir /libjpeg_turbo_perf
RUN mkdir -p /opt/libjpeg-turbo/lib64
COPY ElCapitan_256_by_256.jpg /libjpeg_turbo_perf
COPY libturbojpeg.so /opt/libjpeg-turbo/lib64
COPY ParallelJpegDecodingLJT.java /libjpeg_turbo_perf
COPY turbojpeg.jar /libjpeg_turbo_perf

RUN apt-get update && apt-get install -y openjdk-8-jdk && \
  cd /libjpeg_turbo_perf && javac -cp turbojpeg.jar ParallelJpegDecodingLJT.java

CMD /bin/bash -c "cd /libjpeg_turbo_perf && java -XX:CompileThreshold=1000 -Xmx4096m -Xms4096m -cp turbojpeg.jar:. ParallelJpegDecodingLJT"
