dataset_path=/data2/dataset/xiaochou_colmap_test

python 1video2stream.py --source $dataset_path
python 2convert.py --source $dataset_path
python 3copy_cams.py --source $dataset_path/frame000000 --scene $dataset_path
python 4convert_frames.py --source $dataset_path
python 5sparse2dense.py --source_path $dataset_path

clip_size=10
voxel_size=0.012
python 6geometry_aware_deduplication.py --source $dataset_path --clip_size $clip_size --voxel_size $voxel_size
