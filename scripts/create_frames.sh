# use gpu accelerated container: srun --pty -w devbox5 -G 1 singularity shell /nfs/home/rhotertj/env/mmaction/decord_cuda.sif
source_path="/nfs/home/rhotertj/datasets/hbl/videos"
target_path="/nfs/home/rhotertj/datasets/hbl/frames_2997"
ls $source_path | xargs -n1 -I{} -t sh -c "mkdir -p $target_path/{}.d && ffmpeg -hwaccel cuda -i /nfs/home/rhotertj/datasets/hbl/videos/{} -qscale:v 2 -start_number 0 -vf scale=-1:256 $target_path/{}.d/%06d.jpg"
# Think about -qscale:v 2 for better quality