# Modified from https://github.com/joshua-wu/deepfakes_faceswap/blob/master/model.py
# We believe this github repo is closest to the original code posted by reddit user, deepfakes

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from util.pixel_shuffler import PixelShuffler

# model parameters
IMAGE_SHAPE = (64, 64, 3)
ENCODER_DIM = 1024
OPTIMIZER = ''
LOSS = 'mean_absolute_error'
DEFAULT_MODEL_DIR = {'encoder':'models/encoder.h5', 'decoder_A':'models/decoder_A.h5', 'decoder_B':'models/decoder_B.h5'}

class Autoencoder():
    def __init__(self, model_dir=DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        x = Input(shape=IMAGE_SHAPE)
        
        # set up one encoder and two decoders
        self.encoder = self.Encoder()
        self.decoder_A = self.Decoder()
        self.decoder_B = self.Decoder()
        
        # combine to form two separate autoencoders for training with data of A and data of B respectively
        self.autoencoder_A = Model(x, self.decoder_A(self.encoder(x)))
        self.autoencoder_B = Model(x, self.decoder_B(self.encoder(x)))
        
        self.autoencoder_A.compile(optimizer=optimizer, loss=LOSS)
        self.autoencoder_B.compile(optimizer=optimizer, loss=LOSS)
    
    # helper function for loading previously saved weights
    def load_weights(self):
        try:
            self.encoder.load_weights(self.model_dir['encoder'])
            self.decoder_A.load_weights(self.model_dir['decoder_A'])
            self.decoder_B.load_weights(self.model_dir['decoder_B'])
            print('weights loaded')
        except:
            print('existing model not found, starting fresh...')
    
    # helper function for saving weights
    def save_weights(self):
        self.encoder.save_weights(self.model_dir['encoder'])
        self.decoder_A.save_weights(self.model_dir['decoder_A'])
        self.decoder_B.save_weights(self.model_dir['decoder_B'])

    # a convolution layer block
    def conv(self, filters):
        def block(x):
            x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
            x = LeakyReLU(0.1)(x)
            return x
        return block

    # an upscale block using PixelShuffler layer
    def upscale(self, filters):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
            x = LeakyReLU(0.1)(x)
            x = PixelShuffler()(x)
            return x
        return block

    # set up encoder model
    def Encoder(self):
        input_ = Input(shape=IMAGE_SHAPE)
        x = input_
        x = self.conv(128)(x)
        x = self.conv(256)(x)
        x = self.conv(512)(x)
        x = self.conv(1024)(x)
        x = Dense(ENCODER_DIM)(Flatten()(x)) # the encoded information
        x = Dense(4 * 4 * 1024)(x)
        x = Reshape((4, 4, 1024))(x)
        x = self.upscale(512)(x)
        return Model(input_, x)

    # set up decoder model
    def Decoder(self):
        input_ = Input(shape=(8, 8, 512))
        x = input_
        x = self.upscale(256)(x)
        x = self.upscale(128)(x)
        x = self.upscale(64)(x)
        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        return Model(input_, x)
