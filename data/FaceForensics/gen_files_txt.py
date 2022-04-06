import os

# Constants
ROOT = '/project/6061878/junbinz/FakeMediaDetection/dataset/deepfake/FaceForensics/'

# Paths
save_txt_path = ROOT + '/files.txt'

image_path = ROOT + '/images/'
mask_path = ROOT + '/masks/'
edge_path = ROOT + '/edges/'

# output file
fout = open(save_txt_path, 'w')

# processing
images = os.listdir(image_path)
masks = os.listdir(mask_path)
edges = os.listdir(edge_path)

for i in images:
    if (i not in masks and i not in edges):
        fout.write(os.path.join(image_path, i) + ' None None 0\n')
        continue

    if (i in masks and i in edges):
        fout.write(os.path.join(image_path, i) + ' ' + os.path.join(mask_path, i) + ' ' + os.path.join(edge_path, i) + ' 1\n')
        continue

    print("Error in handling " + i)

fout.close()