# Use same masking work flow as https://github.com/deepfakes/faceswap/blob/master/plugins/Convert_Masked.py

import cv2
import numpy
import glob
import os
from tqdm import tqdm

from util.umeyama import umeyama
from util.face_extractor import *
from util.utils import get_image_paths, load_images, stack_images

from Autoencoder import Autoencoder

'''
A script for swapping faces with trained model.  
'''

# put your parameters here
MODEL_DIR = {'encoder':'models/encoder.h5', 'decoder_A':'models/decoder_A.h5', 'decoder_B':'models/decoder_B.h5'}
INPUT_DIR = 'raw/clooney'
OUTPUT_DIR = 'output'
DIRECTION = 'AtoB' # conversion direction: {'AtoB', 'BtoA'}

# conversion params
size = 256 # aligned face size
input_size = 64 # model input size
crop = slice(48,208) # for cropping (160 x 160) from aligned face
erosion_kernel_size = 10 # erosion inward size for masking output to input image
blur_size = 5 # blur size for masking output to input image

# Setup model and load weights
model = Autoencoder(MODEL_DIR)
model.load_weights()
autoencoder_A = model.autoencoder_A
autoencoder_B = model.autoencoder_B
encoder = model.encoder
decoder_A = model.decoder_A
decoder_B = model.decoder_B

def get_image_mask(face, new_face, mat, input_image, image_size):   
    
    # get mask with transformed input shape
    face_mask = numpy.zeros(input_image.shape,dtype=float)
    face_src = numpy.ones(new_face.shape,dtype=float) 
    cv2.warpAffine( face_src, mat*64, image_size, face_mask, cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT )
    
    # get hull_mask from face landmarks
    hull_mask = numpy.zeros(input_image.shape,dtype=float)
    hull = cv2.convexHull( numpy.array( face.landmarksAsXY() ).reshape((-1,2)).astype(int) ).flatten().reshape( (-1,2) )
    cv2.fillConvexPoly( hull_mask,hull,(1,1,1) )
    
    # combine rect mask and hull mask
    image_mask = ((face_mask*hull_mask))
    
    # apply erosion to mask
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(erosion_kernel_size,erosion_kernel_size))
    image_mask = cv2.erode(image_mask,erosion_kernel,iterations = 1)

    # apply blur to mask
    image_mask = cv2.blur(image_mask,(blur_size,blur_size))
    
    return image_mask

def face_swap(input_image, direction='AtoB'):
    # perform the actual face swap

    image_size = input_image.shape[1], input_image.shape[0]

    # extract face from input image
    facelist = extract_faces(input_image, size)

    # Only consider first face identified
    if len(facelist) > 0:
        face, resized_image = facelist[0]
    else:
        return None
    
    # get alignment matrix
    mat = get_align_mat(face)
    
    resized_face_image = resized_image[crop,crop]
    resized_face_image = cv2.resize( resized_face_image, (input_size, input_size) ) / 255.0

    test_image = numpy.expand_dims(resized_face_image,0)

    # predict faceswap using encoder A or B depends on direction required
    if direction == 'AtoB':
        figure = autoencoder_B.predict(test_image)
    elif direction == 'BtoA':
        figure = autoencoder_A.predict(test_image)
    else:
        print("Invalid direction, 'AtoB' or 'BtoA' only")

    new_face = numpy.clip(numpy.squeeze(figure[0]) * 255.0, 0, 255).astype('uint8')
    
    # get image mask
    image_mask = get_image_mask(face, new_face, mat, input_image, image_size)  
    
    # apply model output face to input image (without mask)
    base_image = numpy.copy( input_image )
    new_image = numpy.copy( input_image )
    cv2.warpAffine( new_face, mat * input_size , image_size, new_image, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )
    
    # use opencv seamless clone to apply new image with the mask on base image    
    unitMask = numpy.clip( image_mask * 365, 0, 255 ).astype(numpy.uint8)
    maxregion = numpy.argwhere(unitMask==255)
    
    if maxregion.size > 0:
        miny,minx = maxregion.min(axis=0)[:2]
        maxy,maxx = maxregion.max(axis=0)[:2]
        lenx = maxx - minx
        leny = maxy - miny
        masky = int(minx+(lenx//2))
        maskx = int(miny+(leny//2))
    output_image = cv2.seamlessClone(new_image.astype(numpy.uint8),base_image.astype(numpy.uint8),(unitMask).astype(numpy.uint8),(masky,maskx) , cv2.NORMAL_CLONE )

    return output_image


def main():
    if not os.path.exists(INPUT_DIR):
        print('INPUT_DIR {} does not exist!'.format(INPUT_DIR))
        return
    
    # make directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # walk through a folder
    for file in tqdm(glob.glob(os.path.join(INPUT_DIR, '*'))):
        filename = os.path.basename(file)
        input_image = cv2.imread(file)
        output_image = face_swap(input_image, DIRECTION)
        if output_image is not None:            
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), output_image)
        else:
            print('No face found in {}'.format(filename))

if __name__ == "__main__":
    main()