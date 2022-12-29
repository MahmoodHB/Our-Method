from tpgan import TPGAN, multipie_gen
from keras.optimizers import SGD 
from keras.optimizers import Adam
import math
import time
import multiprocessing  
from numba import prange
time_total =[]
time1=time.time()


if __name__ == '__main__':
         
    op = 'Adam'
    
    gan = TPGAN(base_filters=64, gpus=1,
                lcnn_extractor_weights='path to lcnn_weights file',
                generator_weights='',
                classifier_weights='',   
                discriminator_weights='')
    
    datagen = multipie_gen.Datagen(dataset_dir='path to dataset', landmarks_dict_file='path to landmarks file', 
                                   datalist_dir='path to datalist pkl file', min_angle=-90, max_angle=90, valid_count="all folders")

    if op == 'Adam':
        optimizer = Adam(lr=0.0001, beta_1=0.9)
    elif op == 'SGD':
        optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
                  
    gan.train_gan(gen_datagen_creator=datagen.get_generator, 
                  gen_train_batch_size=4, #5 - 8
                  gen_valid_batch_size=4,
                  disc_datagen_creator=datagen.get_discriminator_generator, 
                  disc_batch_size=10, #8 - 11
                  disc_gt_shape=gan.discriminator().output_shape[1:3],
                  optimizer=optimizer,
                  gen_steps_per_epoch=10, disc_steps_per_epoch=10,  
                  epochs=300, out_dir='path to output files', out_period=1, is_output_img=True,
                  lr=0.0001, decay=0, lambda_128=1, lambda_Consistency128=0.011, lambda_ip=1e-3, lambda_adv=5e-3, lambda_tv=1e-5, lambda_content=0.05, lambda_parts=3)
				  
    num_trees = 16 #(the number of trees)
    depth = 9 #("5 - 9" each tree uses randomly selected 50% of the input features)
    used_features_rate = 0.5  #("0.1 - 0.5" control the number of features)
    hidden_units = [64, 64]	#(indicating the number of hidden units in each layer)

print ('Time')
time111 = time.time()-time1
print (('Time', time111))
time_total.append (time111)

if __name__ == '__main__':
   pool = multiprocessing.Pool()
   pool.map(1, range(0,10000))
   pool.close()
