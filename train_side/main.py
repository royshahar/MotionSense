import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta
from keras.utils import plot_model
import keras
from data import DataLoader
import models
import time
import cv2
import os
import argparse
import configparser
from ast import literal_eval
import numpy as np


MODEL_FILE_NAME = "3DCNN-{}".format(int(time.time()))

CONFIG_FILE = os.getcwd() + '\\config.cfg'


"""
Function will save the training results into "results_{model_name}.txt" file
"""
def save_history(history, name):
	print (history)
	loss = history.history['loss']
	acc = history.history['acc']
	val_loss = history.history['val_loss']
	val_acc = history.history['val_acc']
	nb_epoch = len(acc)
	
	#iterate through the epochs and write the results
	with open(os.path.join(("models/ " + name + "/"), 'result_{}.txt'.format(name)), 'w') as fp:
		fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
		for i in range(nb_epoch):
			fp.write('{}\t{}\t{}\t{}\t{}\n'.format(i, loss[i], acc[i], val_loss[i], val_acc[i]))


"""
Function will visualize the dataset
"""
def visualize_data(data, train_gen, labels_want):
	if labels_want is None:
		labels_want = data.get_labels_list()

	vids,labels = next(train_gen)
	
	print("batch shape: " + vids.shape)
	
	labels = labels.tolist()
	count = 0
	for vid in vids:
		print(labels_want[labels[count].index(max(labels[count]))])
		for img in vid:
			cv2.imshow("img", img)
			cv2.waitKey()
		count += 1
		

def main(config):
	# Data parameters
	data_dir = config.get('path', 'data_root')
	labels_want = literal_eval(config.get('general', 'labels_want'))
	seq_length = config.getint('general', 'seq_length')
	n_videos = literal_eval(config.get('general', 'n_videos'))
	image_size = literal_eval(config.get('general', 'image_size'))

	# Training parameters
	n_epochs = config.getint('general', 'n_epochs')
	batch_size = config.getint('general', 'batch_size')

	# Load data generators
	data = DataLoader(data_dir, seq_length=seq_length, n_videos=n_videos, labels = labels_want)
	train_gen = data.sequence_generator('train', batch_size, image_size)
	validation_gen = data.sequence_generator('validation', batch_size, image_size)
	
	# --remove the comment from the two lines below in order to visualize the data--
	#visualize_data(data, train_gen, labels_want)
	#exit()
	
	# Calculate the train steps and the validation steps per epoch 
	epoch_steps = int(np.round(np.log(len(data.train_df))))* (len(data.train_df)// batch_size)
	val_steps = int(np.round(np.log(len(data.validation_df))))* (len(data.validation_df)// batch_size)	
	
	# Set the compiling parameters
	metrics = ['accuracy', 'top_k_categorical_accuracy']
	optimizer = Adadelta()
	
	# Load model
	input_shape = ((seq_length//4,) + image_size + (3,))
	n_labels = len(data.get_labels_list())
	model,model_name = models.C3DP_4_LS_1_DE_2_V1(n_labels,input_shape)
		
	model.compile(loss='categorical_crossentropy',
				  optimizer = optimizer,
				  metrics = metrics)
						
	# Create the folder where we will save the model
	path = os.getcwd() + "\\models\\" + model_name
	if not (os.path.exists(path)):
		os.mkdir(path)					
	
	# Write the model architecture into image file
	plot_model(model, to_file = (path + '\\model.png'))
	
	# Print the model architecture
	print(model.summary())
	
	# Create the tensorboard callback
	tensorboard = TensorBoard(log_dir="logs/{}".format(MODEL_FILE_NAME))
	
	# Create the ModelCheckpoint callback
	checkpoint_path = 'models/' + model_name + '/' +	model_name + '-{epoch:03d}-{loss:.3f}.hdf5'
	checkpointer = ModelCheckpoint(filepath=checkpoint_path,
								   verbose=1,
								   save_best_only=True)
	
	# Create the EarlyStopping callback	
	early_stopping = EarlyStopping(monitor='val_loss',
								   min_delta=0,
								   patience=5,
								   verbose=0, 
								   mode='auto')	
			
	# Start the training
	print('Starting training...')
	history = model.fit_generator(generator=train_gen,
								  steps_per_epoch=epoch_steps,
								  epochs=n_epochs,
								  verbose=1,
								  callbacks=[tensorboard,checkpointer,early_stopping],
								  validation_data=validation_gen,
								  validation_steps=val_steps)
	
	# Save the model
	model_path = 'models/' + model_name + '/' +	model_name + '.hdf5'
	model.save((model_path))
	save_history(history,model_name)
	


if __name__ == '__main__':
	# Set tensorflow gpu session parameters
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.9
	session = tf.Session(config = config)
	
	# Parse the arguments from the config file
	parser = argparse.ArgumentParser()
	parser.add_argument("config", type=argparse.FileType('r'))
	args = parser.parse_args([CONFIG_FILE])
	config = configparser.ConfigParser()
	config.read(args.config.name)
	main(config)