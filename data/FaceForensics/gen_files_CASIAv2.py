import os
from pathlib import Path

# Constants
ROOT = '/project/6061878/junbinz/FakeMediaDetection/dataset/manual/ori/CASIAv2/'

# output file
f_train= open(ROOT + '/paths_train_CASIAv2.txt', 'w')
f_test= open(ROOT + '/paths_test_CASIAv2.txt', 'w')
f_val= open(ROOT + '/paths_val_CASIAv2.txt', 'w')

real_image_path = ROOT + '/images/Au/'
fake_image_path = ROOT + '/images/Tp/'
mask_path = ROOT + '/masks/'
edge_path = ROOT + '/edges/'

# processing
real_images = os.listdir(real_image_path)
fake_images = os.listdir(fake_image_path)
masks = os.listdir(mask_path)
edges = os.listdir(edge_path)

n_train = max(len(real_images) // 10 * 8, 1)
n_test = n_train + max(len(real_images) // 10 * 1, 1)
n_val = n_test + max(len(real_images) // 10 * 1, 1)

for i in range(len(real_images)):
    if (i < n_train):
        f_train.write(os.path.join(real_image_path, real_images[i]) + ' None None 0\n')
    elif (i < n_test):
        f_test.write(os.path.join(real_image_path, real_images[i]) + ' None None 0\n')
    elif (i < n_val):
        f_val.write(os.path.join(real_image_path, real_images[i]) + ' None None 0\n')

n_train = max(len(fake_images) // 10 * 8, 1)
n_test = n_train + max(len(fake_images) // 10 * 1, 1)
n_val = n_test + max(len(fake_images) // 10 * 1, 1)
for i in range(len(fake_images)):
    gt_filename = Path(fake_images[i]).stem + '_gt.png'

    if (gt_filename in masks and gt_filename in edges):
        if (i < n_train):
            f_train.write(os.path.join(fake_image_path, fake_images[i]) + ' ' + os.path.join(mask_path, gt_filename) + ' ' + os.path.join(edge_path, gt_filename) + ' 1\n')
        elif (i < n_test):
            f_test.write(os.path.join(fake_image_path, fake_images[i]) + ' ' + os.path.join(mask_path, gt_filename) + ' ' + os.path.join(edge_path, gt_filename) + ' 1\n')
        elif (i < n_val):
            f_val.write(os.path.join(fake_image_path, fake_images[i]) + ' ' + os.path.join(mask_path, gt_filename) + ' ' + os.path.join(edge_path, gt_filename) + ' 1\n')
        continue

    print("Error in handling " + fake_images[i])

f_train.close()
f_test.close()
f_val.close()