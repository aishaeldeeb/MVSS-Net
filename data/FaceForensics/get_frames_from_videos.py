import os
import shutil
from pathlib import Path
import cv2
import math
import numpy as np

# Constants
ROOT = '/project/6061878/junbinz/FakeMediaDetection/dataset/deepfake/FaceForensics/'

REAL_ACTORS_FPS = 1
REAL_YOUTUBE_FPS = 1

FAKE_FACE2FACE_FPS = 1
FAKE_FACESWAP_FPS = 1

MASK_THRESHOLD = 5
EDGE_WIDTH = 5

# Paths
real_actors_path = ROOT + '/extract_frames/original_sequences_test/actors/c23/videos/'
real_youtube_path = ROOT + '/extract_frames/original_sequences_test/youtube/c23/videos/'

fake_face2face_video_path = ROOT + '/extract_frames/manipulated_sequences_test/Face2Face/c23/videos/'
fake_face2face_mask_path = ROOT + '/extract_frames/manipulated_sequences_test/Face2Face/masks/videos/'

fake_faceswap_video_path = ROOT + '/extract_frames/manipulated_sequences_test/FaceSwap/c23/videos/'
fake_faceswap_mask_path = ROOT + '/extract_frames/manipulated_sequences_test/FaceSwap/masks/videos/'

save_image_path = ROOT + '/image/'
save_mask_path = ROOT + '/mask/'
save_edge_path = ROOT + '/edge/'

# functions
def handle_dir_real(video_dir, fps, save_image_path, prefix):
    videos = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]

    for video_filename in videos:
        # get video path
        video_full_path = os.path.join(video_dir, video_filename)

        print("Processing " + video_full_path)

        # handle video
        handle_video_image(video_full_path, fps, save_image_path, prefix)

def handle_dir_fake(video_dir, mask_dir, fps, save_image_path, save_mask_path, save_edge_path, prefix):
    mask_videos = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]

    for video_filename in mask_videos:
        print(video_filename)

        # get mask video path
        mask_video_full_path = os.path.join(mask_dir, video_filename)

        # get corresponding video path
        video_full_path = os.path.join(video_dir, video_filename)
        
        if (not os.path.isfile(video_full_path)):
            print(video_full_path + ' not exists!')
            continue
        
        print("Processing " + video_full_path + " and " + mask_video_full_path)

        # handle video
        handle_video_image(video_full_path, fps, save_image_path, prefix)
        handle_video_mask(mask_video_full_path, fps, save_mask_path, save_edge_path, prefix)

def handle_video_image(video_full_path, fps, save_path, prefix):
    # get filename
    video_filename = Path(video_full_path).stem

    # load videos
    vidcap  = cv2.VideoCapture(video_full_path)
    
    # read the first frame
    success, image = vidcap.read()

    # read frames iterately
    cnt = 0
    while success:
        # save image
        save_filename = prefix + '_' + video_filename.replace('.mp4', '') + '_' + str(cnt) + '.png'
        cv2.imwrite(os.path.join(save_path, save_filename), image)

        # update counter
        cnt += 1

        # read the next frame based on FPS
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(cnt * math.floor(1000 / fps)))
        success,image = vidcap.read()

def handle_video_mask(video_full_path, fps, save_mask_path, save_edge_path, prefix):
    # get filename
    video_filename = Path(video_full_path).stem

    # load videos
    vidcap  = cv2.VideoCapture(video_full_path)
    
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

# Reset save folders
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
handle_dir_real(real_actors_path, REAL_ACTORS_FPS, save_image_path, 'actors')

# real video - youtube
handle_dir_real(real_youtube_path, REAL_YOUTUBE_FPS, save_image_path, 'youtube')

# fake video - face2face
handle_dir_fake(fake_face2face_video_path, fake_face2face_mask_path, FAKE_FACE2FACE_FPS, save_image_path, save_mask_path, save_edge_path, 'face2face')

# fake video - faceswap
handle_dir_fake(fake_faceswap_video_path, fake_faceswap_mask_path, FAKE_FACESWAP_FPS, save_image_path, save_mask_path, save_edge_path, 'faceswap')