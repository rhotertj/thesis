#!/bin/sh
echo "Downloading mvit model (pretrained on Kinetics400)..."
# curl -o models/mvit_b_16x4_k400.pt https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_S_16x4_k400_f302660347.pyth 
curl -o models/mvit_b_16x4.pt https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/MVIT_B_16x4.pyth