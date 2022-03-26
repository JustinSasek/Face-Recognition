from cv2 import VideoCapture, imwrite
import os
import keyboard


WORKING_DIR = '.'
DATASET_NAME = 'eclair-faces'
FILENAME_WIDTH = 16


class DatasetCreator:
    STOP_HOTKEY = 'space'  # hotkey to stop recording

    def __init__(self, working_dir='.', dataset_name='eclair-faces', filename_width=16):
        self.root = os.path.join(WORKING_DIR, DATASET_NAME)
        self.img = os.path.join(self.root, 'img')
        self.idFile = os.path.join(self.root, 'id.txt')
        self.sizeFile = os.path.join(self.root, 'size.txt')
        self.cam = VideoCapture(0)  # TODO: FIND UR WEBCAM !!!!! !  !! !

        self.createDirs()
        with open(self.sizeFile, 'r') as f:
            self.size = int(f.read().strip())

    def mkdir(self, path):
        """
        Makes a directory at the specified path if it doesn't exist.

        Args:
            - path(str): the path to the directory to be created
        """
        try:
            os.mkdir(path)
            print(f'Created {path}')
        except OSError:
            print(f'{path} already exists...')
        # Give read, write, execute permissions
        os.chmod(path, 0o777)

    def mkfile(self, path, init=None):
        """
        Creates a file at a specfied path if it doesn't exist.
        
        Args:
            -
        """
        try:
            with open(path, 'x') as f:
                if init is not None:
                    f.write(str(init))
                print(f'Created {path}')
        except FileExistsError:
            print(f'{path} already exists...')
        # Give read, write, execute permissions
        os.chmod(path, 0o777)

    def createDirs(self):
        """
        Creates directories for the dataset
        """
        self.mkdir(self.root)
        self.mkdir(self.img)
        self.mkfile(self.idFile)
        self.mkfile(self.sizeFile, init='0')  # initialize file with 0 if it does not exist

    def captureData(self, imgId):
        newImgPath = os.path.join(self.img, f'{str(self.size).zfill(FILENAME_WIDTH)}.png')
        result, image = self.cam.read()

        imwrite(str(newImgPath), image)
        os.chmod(newImgPath, 0o777)
        with open(self.idFile, 'a') as f:
            f.write(f'{str(self.size).zfill(FILENAME_WIDTH)}.png {imgId}')
            f.write('\n')

        self.size += 1
        with open(self.sizeFile, 'w') as f:
            f.write(str(self.size))

    def manualLoop(self):
        """
        Continuously takes pictures
        """
        while True:
            imgId = input('Image label (id): ').strip()

            if imgId == '':
                break

            self.captureData(imgId)

    def hotkeyLoop(self):
        """
        Keeps taking pictures until the user presses the stop hotkey and saving them
        """
        while True:
            imgId = input('Image label (id): ').strip()

            if imgId == '':
                break
            print(f'Starting at {str(self.size).zfill(FILENAME_WIDTH)}.png, taking images as fast as possible until {self.STOP_HOTKEY} is pressed...')

            while True:
                # Stop key
                if keyboard.is_pressed(self.STOP_HOTKEY):
                    print(f'Stopped at {str(self.size-1).zfill(FILENAME_WIDTH)}.png')
                    break

                self.captureData(imgId)


if __name__ == "__main__":
    creator = DatasetCreator(WORKING_DIR, DATASET_NAME, FILENAME_WIDTH)
    # creator.hotkeyLoop()
