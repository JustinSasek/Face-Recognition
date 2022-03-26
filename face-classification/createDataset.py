from cv2 import VideoCapture, imwrite
import os
import keyboard
import time

WORKING_DIR = '.'
DATASET_NAME = 'eclair-faces'
FILENAME_WIDTH = 16
WEBCAM_NUM = 0  # TODO: MAKE SURE THIS IS USING THE RIGHT WEBCAM!!!!!!!! Usually 0 is the default webcam on your computer and 1 any external webcam.


class DatasetCreator:
    STOP_HOTKEY = 'space'  # hotkey to stop recording
    DEFAULT_DELAY = 0.5  # default delay for hotkeyloop

    def __init__(self, webcam_num=0, working_dir='.', dataset_name='eclair-faces', filename_width=16):
        self.root = os.path.join(WORKING_DIR, DATASET_NAME)
        self.img = os.path.join(self.root, 'img')
        self.idFile = os.path.join(self.root, 'id.txt')
        self.sizeFile = os.path.join(self.root, 'size.txt')
        self.cam = VideoCapture(webcam_num)  
        # Create directory with image, id, and size files
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
        Creates a file at a specified path if it doesn't exist.
        
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
        # Create path to store image
        newImgPath = os.path.join(self.img, f'{str(self.size).zfill(FILENAME_WIDTH)}.png')
        
        # Capture webcam
        result, image = self.cam.read()
        if result:
            # Save image
            imwrite(str(newImgPath), image)
            os.chmod(newImgPath, 0o777)
            
            # Write the image id to the id file
            with open(self.idFile, 'a') as f:
                f.write(f'{str(self.size).zfill(FILENAME_WIDTH)}.png {imgId}')
                f.write('\n')

            # Update the size file
            self.size += 1
            with open(self.sizeFile, 'w') as f:
                f.write(str(self.size))
        else:
            print('Error: Could not capture image')

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
        delay_input = input(f'Delay between images taken (default {self.DEFAULT_DELAY}): ')
        try:
            hotkey_delay = float(delay_input)
            if hotkey_delay < 0:
                raise ValueError  # cannot have negative delay >:(
        except ValueError:
            hotkey_delay = self.DEFAULT_DELAY

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
                # Capture webcam image
                self.captureData(imgId)
                time.sleep(hotkey_delay)


if __name__ == "__main__":
    creator = DatasetCreator(WEBCAM_NUM, WORKING_DIR, DATASET_NAME, FILENAME_WIDTH)
    creator.hotkeyLoop()
