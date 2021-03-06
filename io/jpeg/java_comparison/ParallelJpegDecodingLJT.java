import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.stream.*;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.Path;

import org.libjpegturbo.turbojpeg.*;

public class ParallelJpegDecodingLJT {

    public static BufferedImage decodeJpeg(String path) {
        try {
            Path p = Paths.get(path);
            byte[] data = Files.readAllBytes(p);
            TJDecompressor tjd = new TJDecompressor(data);
            BufferedImage bi = new BufferedImage(tjd.getWidth(), tjd.getHeight(), BufferedImage.TYPE_INT_RGB);
            tjd.decompress(bi, 0);
            return bi;
        // deal with checked exceptions - lambdas won't work otherwise
        } catch (Exception e) {
            throw new RuntimeException(e.getMessage());
        }
    }


    public static void main(String[] args) throws Exception {
        System.out.println("Running multi-threaded JPEG decode in Java. Please wait.\n");
        final int num = 100000;
        String fileName = "ElCapitan_256_by_256.jpg";
        // create a list of 20k copies of the same file, this is like the file name queue in TF
        List<String> fileList = Stream.generate(() -> fileName).limit(num).collect(Collectors.toList());
        long start = System.currentTimeMillis();
        List<Integer> imageWidths = fileList.
                parallelStream().
                map(path -> decodeJpeg(path).getWidth()).
                collect(Collectors.toList());
        long end = System.currentTimeMillis();
        double totalSec = (end - start) / 1000.0;
        double imagesPerSec = num / totalSec;
        // divide by 2 due to HyperThreading
        double imagesPerSecPerCore = imagesPerSec / (Runtime.getRuntime().availableProcessors() / 2);
        System.out.printf("Running time: %.2f seconds\n", totalSec);        
        System.out.printf("Read %d images\n", imageWidths.size());
        System.out.printf("Images/second: %.2f\n", imagesPerSec);
        System.out.printf("Images/second/core: %.2f\n", imagesPerSecPerCore);
    }
}
