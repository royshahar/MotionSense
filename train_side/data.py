import os
import pandas as pd
import numpy as np
import random
from keras.applications.inception_v3 import preprocess_input
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img
import cv2

class DataLoader():
	"""Class for loading data to match model requirements
	data_dir: directory containing all data
	n_videos: dictionary specifying how many videos you want to use for both
				  train and validation data; loads all videos by default
	labels  : If None, uses all labels. If a list with label names is
			  passed, uses only those specified"""
	def __init__(self, data_dir,seq_length , n_videos = {'train': None, 'validation': None}, labels = None):
		self.data_dir = data_dir
		self.video_dir = data_dir + "\\20bn-jester-v1" #os.path.join(self.data_dir, 'videos')
		self.n_videos = n_videos
		self.seq_length = seq_length

		self.get_labels(labels)

		self.train_df = self.load_video_labels('train')

		self.validation_df = self.load_video_labels('validation')

	
	"""The fucntion gets the labels that we choose train on from total of 27 labels"""
	def get_labels(self, labels):
		path = os.path.join(self.data_dir, 'labels.csv')
		self.labels_df = pd.read_csv(path, names=['label'])
		if labels:
			self.labels_df = self.labels_df[self.labels_df.label.isin(labels)]
		self.labels = [str(label[0]) for label in self.labels_df.values]
		self.n_labels = len(self.labels)

		self.label_to_int = dict(zip(self.labels, range(self.n_labels)))
		self.int_to_label = dict(enumerate(self.labels))
	
	"""The fucntion returns the list of labels we train the model on"""
	def get_labels_list(self):
		return self.labels
	
	"""The fucntion loads the video id of the labels we choose to train on"""
	def load_video_labels(self, data_subset):
		path = os.path.join(self.data_dir, '{}.csv'.format(data_subset))
		df = pd.read_csv(path, sep=';', names=['video_id', 'label'])
		df = df[df.label.isin(self.labels)]
		if self.n_videos[data_subset]:
			df = self.reduce_labels(df, self.n_videos[data_subset])
		return df
	
	
	"""The function takes only the required labels from the pandas dataframe"""
	@staticmethod
	def reduce_labels(df, n_videos):
		grouped = df.groupby('label').count()
		counts = [c[0] for c in grouped.values]
		labels = list(grouped.index)

		# Preserves label distribution
		total_count = sum(counts)
		reduced_counts = [int(count / (total_count / n_videos))
							   for count in counts]

		# Builds a new DataFrame with no more than 'n_videos' rows
		reduced_df = pd.DataFrame()
		for cla, cnt in (zip(labels, reduced_counts)):
			label_df = df[df.label == cla]
			sample = label_df.sort_values('video_id')[:cnt]
			reduced_df = reduced_df.append(sample)

		return reduced_df
		
	
	"""The function adjusts adjust the length of the video"""
	def adjust_sequence_length(self, frame_files):
		
		frame_diff = len(frame_files) - self.seq_length

		if frame_diff == 0:
			# No adjusting needed
			return frame_files
		elif frame_diff > 0:
			# Cuts off first few frames to shorten the video
			return frame_files[frame_diff:]
		else:
			# Repeats the first frame to lengthen video
			return frame_files[:1] * abs(frame_diff) + frame_files

	"""The function returns a random batch of size 'batch_size' of video_ids and labels"""
	@staticmethod
	def random_sample(df, batch_size):
		sample = df.sample(n=batch_size)
		video_ids = list(sample.video_id.values.astype(str))
		labels = list(sample.label.values)

		return video_ids, labels
	
	"""The function preprocess the image to fit the model"""
	@staticmethod
	def preprocess_image(image_array):
		return (image_array / 255. )



	"""The function returns a generator that generates sequences in batches"""
	def sequence_generator(self, split, batch_size, image_size, features = False, model = None):
		if split == 'train':
			df = self.train_df
		if split == 'validation':
			df = self.validation_df
		if split == 'test':
			df = self.test_df

		while True:
			# Load a random batch of video IDs and the corresponding labels
			video_ids, labels = self.random_sample(df, batch_size)
			#Convert labels to one-hot array
			label_ids = [self.label_to_int[label] for label in labels]
			y = to_categorical(label_ids, self.n_labels)

			# Load sequences
			x = []
			for video_id in video_ids:
				path = os.path.join(self.video_dir, video_id)
				if not features:
					sequence = self.build_sequence(path, image_size)
				else:
					sequence = self.build_feature_sequence(path, image_size, model)
				x.append(sequence)
			#for i in range(len(sequence)):
			yield np.array(x), np.array(y)
			
	
	
	
	"""The function builds a sequence that is a 4D numpy array: (frame, height, width, channel)"""
	def build_sequence(self, path, image_size):
		frame_files = os.listdir(path)
		# add sorted, so we can recognize the currect sequence
		frame_files = sorted(frame_files)
		sequence = []

		# Adjust length of sequence to match 'self.seq_length'
		frame_files = self.adjust_sequence_length(frame_files)

		frame_paths = [os.path.join(path, f) for f in frame_files]
		for frame_path in frame_paths[0::4]:
			# Load image into numpy array and preprocess it
			image = load_img(frame_path, target_size=image_size)
			image_array = img_to_array(image)
			#image_array_gray = np.mean(image_array, axis=2)
			#image_array_gray = image_array_gray.reshape(26,39,1)
			image_array = self.preprocess_image(image_array)
			sequence.append(image_array)

		return np.array(sequence)
	

	
	"""he function builds a sequence that is a 1D numpy array: image features"""
	def build_feature_sequence(self, path, image_size, model):
		model.layers.pop()
		model.layers.pop()  # two pops to get to pool layerframe_files = os.listdir(path)
		model.outputs = [model.layers[-1].output]# add sorted, so we can recognize the currect sequence
		model.output_layers = [model.layers[-1]]
		model.layers[-1].outbound_nodes = []
		# Adjust length of sequence to match 'self.seq_length'
		sequence = []
		frame_files = os.listdir(path)
		frame_files = sorted(frame_files)
		frame_files = self.adjust_sequence_length(frame_files)

		frame_paths = [os.path.join(path, f) for f in frame_files]
		for frame_path in frame_paths:
			# Load image into numpy array and preprocess it
			image = load_img(frame_path, target_size=image_size)
			image_array = img_to_array(image)
			image_array = self.preprocess_image(image_array)
			features = model.predict(image_array)
			features = features[0]
			sequence.append(features)
		return np.array(sequence)

	

