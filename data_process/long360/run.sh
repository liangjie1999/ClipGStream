dataset_path=/data8/dataset/longvideos/full_court

python 1video2stream.py --source $dataset_path
python 2copy_cams.py --source $dataset_path/frame000000 --scene $dataset_path
python 3convert_frames.py -s $dataset_path