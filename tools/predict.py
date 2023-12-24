import argparse
from operator import itemgetter
from typing import Optional, Tuple
import cv2
import os
import tabulate
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from mmengine import Config, DictAction

from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.visualization import ActionVisualizer

def write_label_on_video(video_path, pred, label, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, "Predict: {}".format(pred), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Label: {}".format(label), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)
        else:
            break
    cap.release()
    out.release()

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('video_directory', help='directory of video file')
    parser.add_argument('label', help='label file')
    parser.add_argument('out_directory', help='output directory')
    parser.add_argument('category', choices=['train', 'val'], help='is train or not')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    args = parser.parse_args()
    return args

def print_result_table(dict_result, file_path=None):
    import pandas as pd
    try:
        from tabulate import tabulate
    except:
        os.system('pip install tabulate')
        from tabulate import tabulate

    if dict_result['class'] is None:
        dict_result['class'] = np.arange(len(dict_result['total_sample']))
    assert (dict_result['class'] is not None)
    df = pd.DataFrame(dict_result, index=dict_result['class'])
    df["accuracy_per_class"] = df["total_correct_sample"] / df["total_sample"]
    if file_path is None:
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        print("Accuracy: {:.4f}".format(df["total_correct_sample"].sum() / df["total_sample"].sum()))
    else:
        with open(file_path, 'w') as f:
            f.write(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
            f.write("\nAccuracy: {:.4f}".format(df["total_correct_sample"].sum() / df["total_sample"].sum()))

def save_confusion_matrix(preds, true_labels, out_directory, category=None):
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_directory + "{}_confusion_matrix.png".format(category))

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # print(cfg)
    # Build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(cfg, args.checkpoint, device=args.device)
    try:
        with open(args.video_directory + args.category + ".txt") as f:
            annotation_list = f.readlines()
    except:
        print("No annotation file")
        return
    
    map_video_label = dict()
    for annotation in annotation_list:
        annotation = annotation.strip()
        video, label = annotation.split(" ")
        map_video_label[video] = int(label)
    # print(map_video_label)
    # input()

    labels = open(args.label).readlines()
    labels = [x.strip() for x in labels]

    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory, exist_ok=True)

    videos = [video for video in os.listdir(args.video_directory + args.category) if video.endswith('.mp4')]

    if not os.path.exists(args.out_directory + args.category):
        os.makedirs(args.out_directory + args.category, exist_ok=True)

    dict_result = {
        'class': np.arange(len(labels)),
        'label': labels,
        'total_sample': [0] * len(labels),
        'total_correct_sample': [0] * len(labels)
    }

    map_video_pred_label = []
    preds = []
    true_labels = []
    error_videos = []
    for video in tqdm(videos):
        pred_result = inference_recognizer(model, os.path.join(args.video_directory + args.category + '/', video))
        
        pred_scores = pred_result.pred_score.tolist()
        score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
        output_score = score_tuples[:len(labels)]
        score_sorted = sorted(output_score, key=itemgetter(1), reverse=True)
        top5_label = score_sorted[:len(labels)]
        
        results = [(labels[k[0]], k[1]) for k in top5_label]
        pred = results[0][0]
        label_idx = (map_video_label[video])
        # print(label_idx, pred)
        # input()
        print("Video: {}".format(video))
        print("Predict: {}, Label: {}".format(pred, labels[label_idx]))
        # video_result_name = "predict_" + video
        # print("Write result to {}".format(args.out_directory + video_result_name))
        # write_label_on_video(os.path.join(args.video_directory + args.category + '/', video), pred, labels[label_idx], args.out_directory + args.category + '/' + video_result_name)
        if top5_label[0][0] == label_idx:
            dict_result['total_correct_sample'][label_idx] += 1
        dict_result['total_sample'][label_idx] += 1
        # print("top5_label: {}".format(top5_label[0][0]))
        # print("label_idx: {}".format(label_idx))
        # input()
        map_video_pred_label.append("{} {} {}".format(video, top5_label[0][0], label_idx))
        preds.append(top5_label[0][0])
        true_labels.append(label_idx)
    
    save_confusion_matrix(preds, true_labels, args.out_directory + args.category + "/", args.category)
    print_result_table(dict_result, args.out_directory + args.category + "_result.txt")
    file_map_video_pred_label = args.out_directory + args.category + "_map_video_pred_label.txt"
    with open(file_map_video_pred_label, 'w') as f:
        f.write('\n'.join(map_video_pred_label))
        f.write('\n')

if __name__ == '__main__':
    main()