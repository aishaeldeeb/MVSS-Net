import os
import shutil
import cv2
import math
import numpy as np
import random

# Constants
ROOT = '/project/6061878/junbinz/FakeMediaDetection/dataset/deepfake/FaceForensics/'

COMPRESSION = 'c23'

NEW_FPS = 5 # or None if not modified

N_REAL = 6

N_FAKE = 2
RANDOM = False

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
def handle_real_videos(prefix, save_image_path):
    videos_dir = ROOT + '/original_sequences/' + prefix + '/' + COMPRESSION + '/videos/'

    videos = [f for f in os.listdir(videos_dir) if os.path.isfile(os.path.join(videos_dir, f))]

    for video_filename in videos:
        # get video path
        video_full_path = os.path.join(videos_dir, video_filename)

        print("Processing " + video_full_path)

        # load videos
        vidcap  = cv2.VideoCapture(video_full_path)
        
        # randomly get N_REAL real frames
        # ref: https://stackoverflow.com/questions/33650974/opencv-python-read-specific-frame-using-videocapture
        if (NEW_FPS):
            new_frame_cnt = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) // int(vidcap.get(cv2.CAP_PROP_FPS)) * NEW_FPS
        else:
            new_frame_cnt = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        selected = random.sample(range(0, new_frame_cnt - 1), N_REAL)

        # read the first frame
        success, image = vidcap.read()

        cnt = 0
        while success:
            
            if (cnt in selected):
                # save image
                save_filename = prefix + '_' + video_filename.replace('.mp4', '') + '_' + str(cnt) + '.png'
                cv2.imwrite(os.path.join(save_image_path, save_filename), image)

            cnt += 1

            # read the next frame
            if (NEW_FPS):
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(cnt * math.floor(1000 / NEW_FPS)))
            success, image = vidcap.read()

def handle_fake_videos(prefix, save_image_path, save_mask_path, save_edge_path):
    videos_dir = ROOT + '/manipulated_sequences/' + prefix + '/' + COMPRESSION + '/videos/'
    masks_dir = ROOT + '/manipulated_sequences/' + prefix + '/masks/videos/'

    videos = [f for f in os.listdir(videos_dir) if os.path.isfile(os.path.join(masks_dir, f))]

    for video_filename in videos:
        # get video path
        video_full_path = os.path.join(videos_dir, video_filename)

        # get corresponding mask path
        mask_full_path = os.path.join(masks_dir, video_filename)
        
        if (not os.path.isfile(mask_full_path)):
            print(mask_full_path + ' not exists!')
            continue

        print("Processing " + video_full_path + ' and ' + mask_full_path)

        # load videos
        vidcap_v  = cv2.VideoCapture(video_full_path)
        vidcap_m  = cv2.VideoCapture(mask_full_path)
        
        if (RANDOM):
            if (NEW_FPS):
                new_frame_cnt = int(vidcap_v.get(cv2.CAP_PROP_FRAME_COUNT)) // int(vidcap_v.get(cv2.CAP_PROP_FPS)) * NEW_FPS
            else:
                new_frame_cnt = int(vidcap_v.get(cv2.CAP_PROP_FRAME_COUNT))
            selected = random.sample(range(0, new_frame_cnt - 1), N_FAKE)
            
            success_v, image = vidcap_v.read()
            success_m, mask = vidcap_m.read()

            cnt = 0
            while success_v and success_m:
                if (cnt in selected):
                    mask, edge = gen_mask_edge(mask)

                    # save everything
                    save_filename = prefix + '_' + video_filename.replace('.mp4', '') + '_' + str(cnt) + '.png'
                    cv2.imwrite(os.path.join(save_image_path, save_filename), image)
                    cv2.imwrite(os.path.join(save_mask_path, save_filename), mask)
                    cv2.imwrite(os.path.join(save_edge_path, save_filename), edge)
                
                cnt += 1

                # read the next frame
                if (NEW_FPS):
                    vidcap_v.set(cv2.CAP_PROP_POS_MSEC,(cnt * math.floor(1000 / NEW_FPS)))
                    vidcap_m.set(cv2.CAP_PROP_POS_MSEC,(cnt * math.floor(1000 / NEW_FPS)))
                success_v, image = vidcap_v.read()
                success_m, mask = vidcap_m.read()

        else:
            list_areas = []
            list_indices = []
            list_images = []
            list_masks = []
            list_edges = []

            # read the first frame
            success_v, image = vidcap_v.read()
            success_m, mask = vidcap_m.read()

            cnt = 0
            while success_v and success_m:
                mask, edge = gen_mask_edge(mask)

                a = np.sum(mask == 255)

                if (len(list_areas) < N_FAKE):
                    list_areas.append(a)
                    list_indices.append(cnt)
                    list_images.append(image)
                    list_masks.append(mask)
                    list_edges.append(edge)
                else:
                    for i in range(N_FAKE):
                        if (a > list_areas[i]):
                            list_areas[i] = a
                            list_indices[i] = cnt
                            list_images[i] = image
                            list_masks[i] = mask
                            list_edges[i] = edge
                            break

                cnt += 1

                # read the next frame
                if (NEW_FPS):
                    vidcap_v.set(cv2.CAP_PROP_POS_MSEC,(cnt * math.floor(1000 / NEW_FPS)))
                    vidcap_m.set(cv2.CAP_PROP_POS_MSEC,(cnt * math.floor(1000 / NEW_FPS)))
                success_v, image = vidcap_v.read()
                success_m, mask = vidcap_m.read()

            for i in range(len(list_areas)):
                # save everything
                save_filename = prefix + '_' + video_filename.replace('.mp4', '') + '_' + str(list_indices[i]) + '.png'
                cv2.imwrite(os.path.join(save_image_path, save_filename), list_images[i])
                cv2.imwrite(os.path.join(save_mask_path, save_filename), list_masks[i])
                cv2.imwrite(os.path.join(save_edge_path, save_filename), list_edges[i])


def gen_mask_edge(input):
    imgray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(imgray, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

    # get edges
    # ref: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    height, width = mask.shape

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 

    edge = np.zeros((height, width), np.uint8)
    cv2.drawContours(edge, contours, -1, 255, EDGE_WIDTH)

    return mask, edge

# real video - actors
handle_real_videos('actors', save_image_path)

# real video - youtube
handle_real_videos('youtube', save_image_path)

# fake video - Deepfakes
handle_fake_videos('Deepfakes', save_image_path, save_mask_path, save_edge_path)

# fake video - Face2Face
handle_fake_videos('Face2Face', save_image_path, save_mask_path, save_edge_path)

# fake video - FaceSwap
handle_fake_videos('FaceSwap', save_image_path, save_mask_path, save_edge_path)

# fake video - NeuralTextures
handle_fake_videos('NeuralTextures', save_image_path, save_mask_path, save_edge_path)

fout.close()