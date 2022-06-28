import tensorflow as tf
'''
from keras.layers import Input, Add, Maximum, Dense, Activation, BatchNormalization, Conv2D, Conv2DTranspose, Reshape, Flatten, Concatenate, Lambda, MaxPooling2D, ZeroPadding2D, Dropout, AveragePooling2D, Average
from keras import regularizers
'''
from tpgan import LightCNN
import pickle

'''
class LightCNN():
    
    def extractor(self):
        self._extractor = self.build_extractor_29layers_v2(name='extract29v2', block=self._res_block, layers=[1, 2, 3, 4])
        self._extractor.load_weights(self.extractor_weights)
                
        return self._extractor
    
    def _mfm(self, X, name, out_channels, kernel_size=3, strides=1, dense=False):      
        if dense:
            X = Dense(out_channels*2, name = name + '_dense1', kernel_regularizer=regularizers.l2(0.0005))(X)
        else:
            X = Conv2D(out_channels*2, name = name + '_conv2d1', kernel_size=kernel_size, kernel_regularizer=regularizers.l2(0.0005), strides=strides, padding='same')(X)
            
        X = Maximum()([Lambda(lambda x, c: x[..., :c], arguments={'c':out_channels})(X), Lambda(lambda x, c: x[..., c:], arguments={'c':out_channels})(X)])
        
        return X
    
    def _group(self, X, name, in_channels, out_channels, kernel_size, strides):
        
        X = self._mfm(X, name = name + '_mfm1', out_channels=in_channels, kernel_size=1, strides=1, dense=False)
        X = self._mfm(X, name = name + '_mfm2', out_channels=out_channels, kernel_size=kernel_size, strides=strides)
        
        return X
    
    def _res_block(self, X, name, out_channels):
        """
        private func for creating residual block with mfm layers.
        """
        
        X_shortcut = X
        X = self._mfm(X, name = name + '_mfm1', out_channels=out_channels, kernel_size=3, strides=1)
        X = self._mfm(X, name = name + '_mfm2', out_channels=out_channels, kernel_size=3, strides=1)
        X = Add()([X, X_shortcut])
        return X
    
    def _make_layer(self, X, name, block, num_blocks, out_channels):
        """
        private func for creating multiple blocks. block is usualy res_block.
        """
        
        for i in range(0, num_blocks):
            X = block(X, name = name + '_block{}'.format(i), out_channels=out_channels)
        return X
    
    def build_extractor_29layers_v2(self, name, block, layers):
        
        in_img = Input(shape=(*self.in_size_hw, 1))
        
        X = self._mfm(in_img, name = name + '_mfm1', out_channels=48, kernel_size=5, strides=1)
        X = Average()([MaxPooling2D(pool_size=2, padding='same')(X), AveragePooling2D(pool_size=2, padding='same')(X)])
        X = self._make_layer(X, name = name + '_layers1', block=block, num_blocks=layers[0], out_channels=48)
        X = self._group(X, name = name + '_group1', in_channels=48, out_channels=96, kernel_size=3, strides=1)
        X = Average()([MaxPooling2D(pool_size=2, padding='same')(X), AveragePooling2D(pool_size=2, padding='same')(X)])
        X = self._make_layer(X, name = name + '_layers2', block=block, num_blocks=layers[1], out_channels=96)
        X = self._group(X, name = name + '_group2', in_channels=96, out_channels=192, kernel_size=3, strides=1)
        X = Average()([MaxPooling2D(pool_size=2, padding='same')(X), AveragePooling2D(pool_size=2, padding='same')(X)])
        X = self._make_layer(X, name = name + '_layers3', block=block, num_blocks=layers[2], out_channels=192)
        X = self._group(X, name = name + '_group3', in_channels=192, out_channels=128, kernel_size=3, strides=1)
        X = self._make_layer(X, name = name + '_layers4', block=block, num_blocks=layers[3], out_channels=128)
        X = self._group(X, name = name + '_group4', in_channels=128, out_channels=128, kernel_size=3, strides=1)
        feat_map = Average()([MaxPooling2D(pool_size=2, padding='same')(X), AveragePooling2D(pool_size=2, padding='same')(X)])
        feat_vec = Dense(256, name = name + '_dense1', kernel_regularizer=regularizers.l2(0.0005))(Flatten()(feat_map))
        
        ret_extractor = Model(inputs=in_img, outputs=[feat_vec, feat_map], name=name)        
        #ret_extractor.summary()
        
        return ret_extractor
'''
    
if __name__ == "__main__":
    
    lcnn_extractor_weights='E:/tpgan_keras/keras_tpgan/extract29v2_lr0.00010_loss0.997_valacc1.000_epoch1110.hdf5'
    datalist_file ='datalist_test.pkl'
    
    if tf.io.gfile.Exists(datalist_file):
        with open(datalist_file, 'rb') as f:
            datalist = pickle.load(f)
    else:
        print("No datalist file")
        
    lcnn = LightCNN(extractor_type='29v2', extractor_weights=lcnn_extractor_weights)
    lcnn.extractor()
    