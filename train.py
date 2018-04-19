import cv2
import numpy
import os
import glob
import time
import numpy

from util.utils import get_image_paths, load_images, stack_images
from util.training_data import get_training_data

from Autoencoder import Autoencoder

# change parameters here
MODEL_DIR = {'encoder':'models/encoder.h5', 'decoder_A':'models/decoder_A.h5', 'decoder_B':'models/decoder_B.h5'}
MODEL_BASE_DIR = 'models'
TRAIN_LOG_PATH = 'models/train_log.txt'
IMAGE_A_DIR = "data/clooney"
IMAGE_B_DIR = "data/cage"  
SHOW_PREVIEW = True
SAVE_PREVIEW = True
FIXED_PREVIEW = True
FIXED_PREVIEW_PATH = 'models/previews'

# hack for FIXED_PREVIEW
def save_fixed_preview_img(images_A, images_B):
    numpy.save(os.path.join(FIXED_PREVIEW_PATH, 'A_preview.npy'), images_A)
    numpy.save(os.path.join(FIXED_PREVIEW_PATH, 'B_preview.npy'), images_B)
    print('saved new fix preview images')

# hack for FIXED_PREVIEW
def load_fixed_preview_img(images_A, images_B):
    if not os.path.exists(FIXED_PREVIEW_PATH):
        os.makedirs(FIXED_PREVIEW_PATH)
        save_fixed_preview_img(images_A, images_B)
        return images_A, images_B
    
    try:
        load_images_A = numpy.load(os.path.join(FIXED_PREVIEW_PATH, 'A_preview.npy'))
        load_images_B = numpy.load(os.path.join(FIXED_PREVIEW_PATH, 'B_preview.npy'))
        print('loaded fix preview image')
    except:
        print('fix preview images not found, use new images')
        save_fixed_preview_img(images_A, images_B)
        return images_A, images_B

    return load_images_A, load_images_B

def main():
    model = Autoencoder(MODEL_DIR)
    model.load_weights()
    autoencoder_A = model.autoencoder_A
    autoencoder_B = model.autoencoder_B
    
    print('loading training data...')
    images_A = get_image_paths( IMAGE_A_DIR )
    images_B = get_image_paths( IMAGE_B_DIR )
    images_A = load_images( images_A ) / 255.0
    images_B = load_images( images_B ) / 255.0
    
    # hack for FIXED_PREVIEW
    fixed_preview_img_initialized = False

    print( "press 'q' to stop training and save model" )
    
    # open file for log
    if not os.path.exists(os.path.dirname(TRAIN_LOG_PATH)):
        os.mkdirs(os.path.dirname(TRAIN_LOG_PATH))

    with open(TRAIN_LOG_PATH, 'a') as f:
        for epoch in range(1000000):
            batch_size = 64
            warped_A, target_A = get_training_data( images_A, batch_size )
            warped_B, target_B = get_training_data( images_B, batch_size )

            # hack for FIXED_PREVIEW
            if FIXED_PREVIEW and not fixed_preview_img_initialized:
                test_A, test_B = load_fixed_preview_img(target_A[0:14], target_B[0:14])
                fixed_preview_img_initialized = True

            loss_A = autoencoder_A.train_on_batch( warped_A, target_A )
            loss_B = autoencoder_B.train_on_batch( warped_B, target_B )
            msg = "time: {:.2f} epoch: {} loss_A: {} loss_B: {}".format(time.time(), epoch, loss_A, loss_B)
            print(msg)
            f.write(msg+'\n')
            f.flush()

            if epoch % 100 == 0:
                model.save_weights()
                
                # hack for FIXED_PREVIEW
                if not FIXED_PREVIEW:
                    test_A = target_A[0:14]
                    test_B = target_B[0:14]                    

                figure_A = numpy.stack([
                    test_A,
                    autoencoder_A.predict( test_A ),
                    autoencoder_B.predict( test_A ),
                    ], axis=1 )
                figure_B = numpy.stack([
                    test_B,
                    autoencoder_B.predict( test_B ),
                    autoencoder_A.predict( test_B ),
                    ], axis=1 )

                figure = numpy.concatenate( [ figure_A, figure_B ], axis=0 )
                figure = figure.reshape( (4,7) + figure.shape[1:] )
                figure = stack_images( figure )

                figure = numpy.clip( figure * 255, 0, 255 ).astype('uint8')
                if SHOW_PREVIEW:
                    cv2.imshow( "", figure )
                if SAVE_PREVIEW:
                    cv2.imwrite(os.path.join(MODEL_BASE_DIR, 'model_preview_epoch_{}.jpg'.format(epoch)), figure)

            key = cv2.waitKey(1)
            if key == ord('q'):
                print('saving weights...')
                model.save_weights()
                exit()

if __name__ == '__main__':
    main()