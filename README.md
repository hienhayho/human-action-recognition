# HUMAN ACTION RECOGNITION (VIDEO MERL SHOPPING)

This is the code of mine in `Datathon 2023 Challenge` based on [`mmaction2`](https://github.com/open-mmlab/mmaction2), you can see the demo here: [Demo](https://youtu.be/5HXY9q-BNh8)

## Prepare dataset:
In this repo, I use format of `kinetics` datasets. Please follow the folder structure below:

    data
    ├── train
    |    ├── train_00001.mp4
    |    ├── train_00002.mp4
    |    ├── ... 
    ├── val
    |    ├──val_00001.mp4
    |    ├──val_00002.mp4
    |    ├── ...             
    ├── train.txt                     
    ├── val.txt
    ├── label.txt             

- About each video in `train` and `val` folder, please notice that its frames should be a mutiple of 25 to prevent unexpected errors.
- Format of `train.txt` and `val.txt`, the following number refer to the label of the video.

```
train_00001.mp4 5
train_00002.mp4 4
...
```
- `label.txt` is the file containing all your label actions in `train.txt` and `val.txt`. For example, the competition provided *6* actions and in `label.txt` it should be:
```
0
1
2
3
4
5
```
> Note: If your data similar to [`MERL Shopping dataset`](https://paperswithcode.com/dataset/merl-shopping) which has crop video and `.mat` label file, you can use `tools/prepare_data_2.py` to get the data structure above.

More further information about the kinetics dataset, you can refer to [mmaction2](https://github.com/open-mmlab/mmaction2) guide: [Prepare Datasets](https://mmaction2.readthedocs.io/en/latest/user_guides/prepare_dataset.html)

## Set up Environment
**1.** You can refer to [mmaction2 installation](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html). (But sometimes it costs much time for setting up from scratch)

**2.** I have set up and saved this `image` to `docker hub` and you can use this instead of setting up enviroment from scratch.

- Require **docker** to run this.

```bash
# Pull the image
docker pull hienhayho/mmaction2

# Start the docker container
docker run -d --gpus all --shm-size=4G -it -v path/to/your/folder:path/to/your/folder --name mmaction2 hienhayho/mmaction2:latest bash

# Execute your container
docker exec -it mmaction2 bash

# Activate venv
conda activate openmmlab
```

## Training
**1. About the pretrained model, you can download it from**: [here](https://drive.google.com/file/d/1Z_D5IcJx35gMHuwZYnNvZe7_N6dzPVPO/view?usp=sharing)

**2. In config file:**  `mvit/mvit-base-p244_32x3x1_kinetics400-rgb.py`. Please set these values:

```
# Your local dataset path
ann_file_test = ...   # val.txt
ann_file_train = ...  # train.txt
ann_file_val = ...    # val.txt
data_root = ...       # train/
data_root_val = ...   # val/

...

#Set downloaded pretrained model path 
load_from = ...
```

> Note: You can try with different config files in `config` provided by `mmaction2`

**3. Training**
```python3
CUDA_VISIBLE_DEVICES=0 python3 \
    tools/train.py \
    mvit/mvit-base-p244_32x3x1_kinetics400-rgb.py \
    --work-dir train_mvit/
```
> Note: If you encounter errors about the video data, please use [`decord`](https://github.com/dmlc/decord) to try loading videos and delete error videos.

## Video Demo
Use `demo/long_video_demo.py` to inference on a video.

For example:
```python3
CUDA_VISIBLE_DEVICES=0 python3 \
    demo/long_video_demo.py \
    your_path_to_config_file \
    your_path_to_check_point \
    demo/9_3_crop.mp4 \
    your_path_to_label_file \
    video_demo/demo.mp4 \
    --batch-size 4
```

> For more info about the provided arguments, please refer to `demo/long_video_demo.py`