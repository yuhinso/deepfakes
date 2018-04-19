from util.face_extractor import extract_faces
import glob
import os
import cv2
from tqdm import tqdm

'''
A script for preparing training data (face images) by extracting and aligning the face in input images.

Assumption:
* Only the first face found in the image will be saved
* Please double check your output data before training the model
'''

# change your directories here
INPUT_DIR = 'raw/clooney_interview'
OUTPUT_DIR = 'data/clooney'

def main():
    if not os.path.exists(INPUT_DIR):
        print('INPUT_DIR {} does not exist!'.format(INPUT_DIR))
        return
    
    # make directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print('OUTPUT_DIR: {}'.format(OUTPUT_DIR))
    print('Start extracting faces...')

    # walk through a folder
    for file in tqdm(glob.glob(os.path.join(INPUT_DIR, '*'))):
        filename = os.path.basename(file)
        input_image = cv2.imread(file)
        # extract faces
        face_list = extract_faces(input_image, 256)
        if len(face_list)>0:
            face, resized_image = face_list[0]
        # save the first face found
        cv2.imwrite(os.path.join(OUTPUT_DIR, '{}.png'.format(filename)), resized_image)

if __name__ == '__main__':
    main()