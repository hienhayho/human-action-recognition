import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Delete error video")
    parser.add_argument('video_directory', help='directory of video file needed to be deleted')
    parser.add_argument('file_error_path', help='file path of error video')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.file_error_path) as f:
        error_list = f.readlines()
    for i in range(len(error_list)):
        error_list[i] = error_list[i].strip()
        error_list[i] = error_list[i].replace("\n", "")
    
    for video in os.listdir(args.video_directory):
        if video in error_list:
            # print("Delete {}".format(video))
            os.remove(args.video_directory + video)
            print("Delete {}".format(video)) 
    # print(error_list)

if __name__ == '__main__':
    main()