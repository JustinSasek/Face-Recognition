from cv2 import VideoCapture, imwrite
import os


WORKING_DIR = '.'
DATASET_NAME = 'eclair-faces'

root = os.path.join(WORKING_DIR, DATASET_NAME)
img = os.path.join(root, 'img')
idFile = os.path.join(root, 'id.txt')
sizeFile = os.path.join(root, 'size.txt')


def mkdir(path):
    try:
        os.mkdir(path)
        print(f'Created {path}')
    except OSError:
        print(f'{path} already exists...')



def createDirs():
    mkdir(root)
    mkdir(img)

    try:
        with open(idFile, 'x') as f:
            print(f'Created {idFile}')
    except FileExistsError:
        print(f'{idFile} already exists...')

    try:
        with open(sizeFile, 'x') as f:
            f.write('0')
            print(f'Created {sizeFile}')
    except FileExistsError:
        print(f'{sizeFile} already exists...')


def cameraLoop():
    cam = VideoCapture(1)
    with open(sizeFile, 'r') as f:
        size = int(f.read().strip())

    while True:
        imgId = input()

        if imgId == '':
            break

        newImgPath = os.path.join(img, f'{str(size).zfill(16)}.png')
        result, image = cam.read()

        imwrite(str(newImgPath), image)
        with open(idFile, 'a') as f:
            f.write(f'{str(size).zfill(16)}.png {imgId.strip()}')
            f.write('\n')

        size += 1
        with open(sizeFile, 'w') as f:
            f.write(str(size))


createDirs()

cameraLoop()
