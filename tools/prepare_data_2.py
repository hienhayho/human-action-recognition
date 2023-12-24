import os
from moviepy.editor import VideoFileClip
import numpy as np
from tqdm import tqdm
import random
from scipy.io import loadmat

def cut_video_by_frames(clip, output_path, start_frame, end_frame, fps=25):
    subclip = clip.subclip(start_frame // clip.fps, end_frame // clip.fps)
    subclip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps)

def print_result_table(dict_result_train, dict_result_val, file_path=None):
    try:
        import pandas as pd
    except:
        os.system('pip install pandas')
        import pandas as pd
    try:
        from tabulate import tabulate
    except:
        os.system('pip install tabulate')
        from tabulate import tabulate

    if dict_result_train['class'] is None:
        dict_result_train['class'] = np.arange(len(dict_result_train['total_sample']))
    if dict_result_val['class'] is None:
        dict_result_val['class'] = np.arange(len(dict_result_val['total_sample']))
    # assert (dict_result['class'] is not None)
    df_train = pd.DataFrame(dict_result_train, index=dict_result_train['class'])
    df_val = pd.DataFrame(dict_result_val, index=dict_result_val['class'])
    if file_path is None:
        print(tabulate(df_train, headers='keys', tablefmt='psql', showindex=False))
        print("\n")
        print(tabulate(df_val, headers='keys', tablefmt='psql', showindex=False))
    else:
        with open(file_path, 'w') as f:
            f.write(tabulate(df_train, headers='keys', tablefmt='psql', showindex=False))
            f.write("\n")
            f.write(tabulate(df_val, headers='keys', tablefmt='psql', showindex=False))

def main():
    video_raw_path = "Videos_MERL_Shopping_Dataset/"
    label_raw_path = "Labels_MERL_Shopping_Dataset/"
    result_train_file_path = "data_preprocessed_full/train.txt"
    result_val_file_path = "data_preprocessed_full/val.txt"
    video_train_save_path = "data_preprocessed_full/train/"
    video_val_save_path = "data_preprocessed_full/val/"
    video_path = [video for video in os.listdir(video_raw_path) if video.endswith(".mp4")]
    video_train_path_list = []
    video_val_path_list = []

    train_idx = 1
    val_idx = 1

    dict_train_sample = {
        "class": np.arange(6),
        "total_sample": np.zeros(6)
    }
    dict_val_sample = {
        "class": np.arange(6),
        "total_sample": np.zeros(6)
    }

    for video in tqdm(video_path):
        video_name = video.split(".")[0]
        # print(video_name)
        label_name = video_name.replace("crop", "label") + ".mat"
        data = loadmat(os.path.join(label_raw_path, label_name))
        min_frame_not_class_0 = 100000000
        print("Doing in video, label: ", video, label_name)
        for i in range(len(data["tlabs"])):
            if len(data["tlabs"][i]) == 0: # check empty
                continue
            for values in enumerate(data["tlabs"][i]):
                for j, value in enumerate(values[1]):
                    # Split each [start_frame, end_frame] into 4 parts: train_1, val_1, train_2, val_2
                    start_frame, end_frame = int(value[0]), int(value[1])
                    min_frame_not_class_0 = min(min_frame_not_class_0, start_frame)
                    if end_frame - start_frame + 1 < 75:
                        continue
                    # frame_per_shot = (end_frame - start_frame) // 10
                    # end_train_frame_1 = start_frame + frame_per_shot * 3
                    # end_val_frame_1 = end_train_frame_1 + frame_per_shot * 2
                    # end_train_frame_2 = end_val_frame_1 + frame_per_shot * 3
                    # end_val_frame_2 = end_frame
                    # frame_per_shot = 24
                    span = (end_frame - start_frame) // 25
                    train_span = span // 2
                    val_span = span - train_span
                    end_train_frame_1 = start_frame + train_span * 25
                    end_val_frame_1 = end_train_frame_1 + val_span * 25
                    clip = VideoFileClip(os.path.join(video_raw_path, video))
                    file_train_1 = "train_{:05d}".format(train_idx) + ".mp4"
                    train_idx += 1
                    file_val_1 = "val_{:05d}".format(val_idx) + ".mp4"
                    val_idx += 1

                    video_train_path_list.append(file_train_1 + " {}".format(i + 1))
                    dict_train_sample["total_sample"][i + 1] += 1
                    dict_val_sample["total_sample"][i + 1] += 1
                    video_val_path_list.append(file_val_1 + " {}".format(i + 1))
                    #save train 1
                    cut_video_by_frames(clip, os.path.join(video_train_save_path, file_train_1), start_frame, end_train_frame_1)
                    #save val 1
                    cut_video_by_frames(clip, os.path.join(video_val_save_path, file_val_1), end_train_frame_1, end_val_frame_1)
        if min_frame_not_class_0 < 75:
            continue
        end_frame_class_0 = min(500, min_frame_not_class_0)
        clip = VideoFileClip(os.path.join(video_raw_path, video))
        span = end_frame_class_0 // 25
        train_span = span // 2
        val_span = span - train_span
        end_train_frame_1 = train_span * 25
        end_val_frame_1 = end_train_frame_1 + val_span * 25
        file_train_1 = "train_{:05d}".format(train_idx) + ".mp4"
        train_idx += 1
        file_val_1 = "val_{:05d}".format(val_idx) + ".mp4"
        val_idx += 1
        # train 1
        video_train_path_list.append(file_train_1 + "{}".format(0))
        dict_train_sample["total_sample"][0] += 1
        dict_val_sample["total_sample"][0] += 1
        video_val_path_list.append(file_val_1 + " {}".format(0))
        # save train 1
        cut_video_by_frames(clip, os.path.join(video_train_save_path, file_train_1), 0, end_train_frame_1)
        # save val 1
        cut_video_by_frames(clip, os.path.join(video_val_save_path, file_val_1), end_train_frame_1, end_val_frame_1)
    
    random.shuffle(video_train_path_list)
    random.shuffle(video_val_path_list)
    with open(result_train_file_path, "w") as f:
        for video in video_train_path_list:
            f.write(video + "\n")
    with open(result_val_file_path, "w") as f:
        for video in video_val_path_list:
            f.write(video + "\n")
    print_result_table(dict_train_sample, dict_val_sample, "data_preprocessed_full/total_sample.txt")

if __name__ == "__main__":
    main()
        