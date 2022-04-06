import os
import shutil
from pathlib import Path
import cv2
import math
import numpy as np

# Constants
ROOT = '/project/6061878/junbinz/FakeMediaDetection/dataset/deepfake/FaceForensics/'

COMPRESSION = 'c23'

MASK_THRESHOLD = 5
EDGE_WIDTH = 5

# Paths
save_image_path = ROOT + '/image/'
save_mask_path = ROOT + '/mask/'
save_edge_path = ROOT + '/edge/'

save_txt_path = ROOT + '/files.txt'

# global file
fout = open(save_txt_path, "w")

# functions
def handle_video_dir(prefix, real, fps, save_image_path):
    video_dir = ROOT + ('/original_sequences/' if real else '/manipulated_sequences/') + prefix + '/' + COMPRESSION + '/videos/'

    videos = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]

    for video_filename in videos:
        # get video path
        video_full_path = os.path.join(video_dir, video_filename)

        print("Processing " + video_full_path)

        # load videos
        vidcap  = cv2.VideoCapture(video_full_path)
        
        # read the first frame
        success, image = vidcap.read()

        # read frames iterately
        cnt = 0
        while success:
            # save image
            save_filename = prefix + '_' + video_filename.replace('.mp4', '') + '_' + str(cnt) + '.png'
            cv2.imwrite(os.path.join(save_image_path, save_filename), image)

            # update counter
            cnt += 1

            # read the next frame based on FPS
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(cnt * math.floor(1000 / fps)))
            success,image = vidcap.read()

def handle_mask_dir(prefix, fps, save_mask_path, save_edge_path):
    mask_dir = ROOT + '/manipulated_sequences/' + prefix + '/masks/videos/'

    mask_videos = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]

    for video_filename in mask_videos:
        # get mask video path
        mask_video_full_path = os.path.join(mask_dir, video_filename)

        print("Processing " + mask_video_full_path)

        # load videos
        vidcap  = cv2.VideoCapture(mask_video_full_path)
        
        # read the first frame
        success, image = vidcap.read()

        # read frames iterately
        cnt = 0
        while success:
            # convert image to mask
            # ref: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
            imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(imgray, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

            # get edges
            # ref: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
            height, width = mask.shape

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 

            edge = np.zeros((height, width), np.uint8)
            cv2.drawContours(edge, contours, -1, 255, EDGE_WIDTH)

            # save mask and edge
            save_filename = prefix + '_' + video_filename.replace('.mp4', '') + '_' + str(cnt) + '.png'
            cv2.imwrite(os.path.join(save_mask_path, save_filename), mask)
            cv2.imwrite(os.path.join(save_edge_path, save_filename), edge)

            # update counter
            cnt += 1

            # read the next frame based on FPS
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(cnt * math.floor(1000 / fps)))
            success,image = vidcap.read()

### main
# reset save folders
if os.path.isdir(save_image_path):
    shutil.rmtree(save_image_path)

if os.path.isdir(save_mask_path):
    shutil.rmtree(save_mask_path)

if os.path.isdir(save_edge_path):
    shutil.rmtree(save_edge_path)

os.makedirs(save_image_path)
os.makedirs(save_mask_path)
os.makedirs(save_edge_path)

# real video - actors
handle_video_dir('actors', True, 0.2, save_image_path)

# real video - youtube
handle_video_dir('youtube', True, 0.2, save_image_path)

# fake video - Deepfakes
handle_video_dir('Deepfakes', False, 0.2, save_image_path)
handle_mask_dir('Deepfakes', 0.2, save_mask_path, save_edge_path)

# fake video - Face2Face
handle_video_dir('Face2Face', False, 0.2, save_image_path)
handle_mask_dir('Face2Face', 0.2, save_mask_path, save_edge_path)

# fake video - FaceSwap
handle_video_dir('FaceSwap', False, 0.2, save_image_path)
handle_mask_dir('FaceSwap', 0.2, save_mask_path, save_edge_path)

# fake video - NeuralTextures
handle_video_dir('NeuralTextures', False, 0.2, save_image_path)
handle_mask_dir('NeuralTextures', 0.2, save_mask_path, save_edge_path)

fout.close()