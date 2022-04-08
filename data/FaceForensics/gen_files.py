import os

# Constants
ROOT = '/project/6061878/junbinz/FakeMediaDetection/dataset/deepfake/FaceForensics/'

SETS = ['actors', 'youtube', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

# output file
f_train= open(ROOT + '/paths_train.txt', 'w')
f_test= open(ROOT + '/paths_test.txt', 'w')
f_val= open(ROOT + '/paths_val.txt', 'w')

# processing
for s in SETS:
    image_path = ROOT + '/image/' + s
    mask_path = ROOT + '/mask/' + s
    edge_path = ROOT + '/edge/' + s

    images = os.listdir(image_path)
    masks = os.listdir(mask_path)
    edges = os.listdir(edge_path)

    n_train = max(len(images) // 10 * 8, 1)
    n_test = n_train + max(len(images) // 10 * 1, 1)
    n_val = n_test + max(len(images) // 10 * 1, 1)

    for i in range(len(images)):
        if (images[i] not in masks and images[i] not in edges):
            if (i < n_train):
                f_train.write(os.path.join(image_path, images[i]) + ' None None 0\n')
            elif (i < n_test):
                f_test.write(os.path.join(image_path, images[i]) + ' None None 0\n')
            elif (i < n_val):
                f_val.write(os.path.join(image_path, images[i]) + ' None None 0\n')
            continue

        if (images[i] in masks and images[i] in edges):
            if (i < n_train):
                f_train.write(os.path.join(image_path, images[i]) + ' ' + os.path.join(mask_path, images[i]) + ' ' + os.path.join(edge_path, images[i]) + ' 1\n')
            elif (i < n_test):
                f_test.write(os.path.join(image_path, images[i]) + ' ' + os.path.join(mask_path, images[i]) + ' ' + os.path.join(edge_path, images[i]) + ' 1\n')
            elif (i < n_val):
                f_val.write(os.path.join(image_path, images[i]) + ' ' + os.path.join(mask_path, images[i]) + ' ' + os.path.join(edge_path, images[i]) + ' 1\n')
            continue

        

        print("Error in handling " + images[i])

f_train.close()
f_test.close()
f_val.close()