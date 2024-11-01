docker run --rm -it \
  --gpus=all \
  --ipc=host \
  -v /dataset/ssd_100k/:/datasets/open-images-v6-mlperf \
  mlperf/single_stage_detector bash 

