# Custom Dataset Processing Pipeline Description
## Usage

1. Place multi-view video files in the same directory
|---
|---view000.mp4
|---view001.mp4
|   ...
|---view035.mp4  

2. Set the following parameters in `run.sh`:
   - `source_path`: directory from step 1
   - `clip_size`: clip dimension (M)
   - `voxel_size`: point cloud downsampling ratio

3. Execute the entire pipeline by running:
```bash
./run.sh
```

## Script Descriptions

1. `1video2stream.py`  
   - Function: Convert video files into image sequence streams
   - Parameters: `--source` specifies the dataset path

2. `2convert.py`  
   - Function: Perform data format conversion
   - Parameters: `--source` specifies the dataset path

3. `3copy_cams.py`  
   - Function: Copy camera parameter files
   - Parameters:
     - `--source` specifies the source frame path
     - `--scene` specifies the target scene path

4. `4convert_frames.py`  
   - Function: Image undistortion
   - Parameters: `--source` specifies the dataset path

5. `5sparse2dense.py`  
   - Function: Extract dense point clouds by using COLMAP
   - Parameters:
     - `--source_path` specifies the dataset path

6. `6geometry_aware_deduplication.py`  
   - Function: Geometry-aware deduplication processing
   - Parameters:
     - `--source` specifies the dataset path
     - `--clip_size` sets the clip size (default: 10)
     - `--voxel_size` sets the voxel size (default: 0.012)

