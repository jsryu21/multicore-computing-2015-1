#!/bin/bash
./plot_data.py input centroid.point data.point input.png
./plot_data.py result final_centroid_pthread.point data.point result_pthread.class result_pthread.png
./plot_data.py result final_centroid_seq.point data.point result_seq.class result_seq.png
