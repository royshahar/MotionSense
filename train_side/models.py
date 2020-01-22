from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam, Nadam
from keras.layers import Dense, Dropout, Flatten, TimeDistributed, LSTM, Input, RepeatVector
from keras.layers.convolutional import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.applications.vgg16 import VGG16
from keras.applications import InceptionV3

#C3D - 3d convolution layer
#C3DP - 3d convolution layer + max pooling
#DE - dense layer
#LS - lstm layer
#C2D - 2d convolution layer

def C2D(n_labels, input_shape):
	model_name = 'C2D'
	model= Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(26,39,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(GlobalAveragePooling2D())
	#Dense Block 1 + Dropout
	model.add(Dense(512, activation='sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(n_labels, activation='softmax'))
	return (model,model_name)

def C2D_LS(n_labels, input_shape):
	model_name = 'C2D_LS'
	model= Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=input_shape[1:], activation='relu'))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(RepeatVector(input_shape[0]))
	model.add(LSTM(128, return_sequences=True, input_shape=(input_shape), activation='relu'))
	model.add(LSTM(128, return_sequences=True, activation='relu'))
	model.add(LSTM(256))
	model.add(Dropout(0.25))
	model.add(Dense(37))
	return (model,model_name)
	

def C3DP_4_DE_3_V1(n_labels, input_shape):
	#image size (26,39) and seq length 32
	# Tunable parameters
	model_name = 'C3DP_4_DE_3_V1'
	kernel_size = (3, 3, 3)
	strides = (1, 1, 1)
	extra_conv_blocks = 1

	model = Sequential()

	# 3D Conv Block 1 + Max Pooling
	model.add(Conv3D(32, (7,7,5), strides=strides, activation='relu',
				padding='same', input_shape=input_shape))
	model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

	# 3D Conv Block 2 + Max Pooling
	model.add(Conv3D(64, (5,5,3), strides=strides, activation='relu',
					 padding='same'), input_shape = (32, 13, 19, 3))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

	# 3D Conv Block 3 + Max Pooling
	model.add(Conv3D(128, (5,5,3), strides=strides, activation='relu',
					 padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

	# 3D Conv Block 4 + Max Pooling + Dropout
	model.add(Conv3D(128, (3,5,3), strides=strides, activation='relu',
					 padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
	model.add(Dropout(0.25))

	
	model.add(Flatten())
	
	# Dense Block 1
	model.add(Dense(256, activation='relu'))
	
	# Dense Block 2 + Dropout
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.2))
	
	# Dense Block 3
	model.add(Dense(n_labels, activation='softmax'))

	return (model,model_name)
	
	
def C3D_2_C3DP_2_DE_2(n_labels, input_shape):
		#image size (26,39) and seq length 32
		# Tunable parameters
		model_name = 'C3D_2_C3DP_2_DE_2'
		strides = (1, 1, 1)
		model = Sequential()
		
		# 3D Conv Block 1
		model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=strides,input_shape=input_shape, 
					border_mode='same', activation='relu'))
		
		# 3D Conv Block 2 + Max Pooling + Dropout
		model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=strides,padding='same', activation='softmax'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
		model.add(Dropout(0.25))

		# 3D Conv Block 3
		model.add(Conv3D(64, kernel_size=(3, 3, 3),strides=strides, padding='same', activation='relu'))
		
		# 3D Conv Block 4 + Max Pooling + Dropout
		model.add(Conv3D(64, kernel_size=(3, 3, 3),strides=strides, padding='same', activation='softmax'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
		model.add(Dropout(0.25))

		model.add(Flatten())
		
		#Dense Block 1 + Dropout
		model.add(Dense(512, activation='sigmoid'))
		model.add(Dropout(0.5))
		
		#Dense Block 2
		model.add(Dense(n_labels, activation='softmax'))

		return (model,model_name)	
		
		
		
	
def C3DP_4_DE_3_V2(n_labels, input_shape):
		"""See: 'https://arxiv.org/pdf/1412.0767.pdf' """
		#image size (26,39) and seq length 32
		#val loss = 0.25
		# Tunable parameters
		model_name = 'C3DP_4_DE_3_V2'
		kernel_size = (3, 3, 3)
		strides = (1, 1, 1)
		extra_conv_blocks = 1

		model = Sequential()
		
		# 3D Conv Block 1 + Max Pooling
		model.add(Conv3D(32, kernel_size, strides=strides, activation='relu',
						 padding='same', input_shape=input_shape))
		model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

		# 3D Conv Block 2 + Max Pooling
		model.add(Conv3D(64, kernel_size, strides=strides, activation='relu',
						 padding='same'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

		# 3D Conv Block 3 + Max Pooling
		model.add(Conv3D(128, kernel_size, strides=strides, activation='relu',
						 padding='same'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

		# 3D Conv Block 4 + Max Pooling
		model.add(Conv3D(128, kernel_size, strides=strides, activation='relu',
						 padding='same'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
		
		
		model.add(Flatten())
		
		# Dense Block 1 + Dropout
		model.add(Dense(256, activation='relu'))
		model.add(Dropout(0.25))
		
		#Dense Block 2 + Dropout
		model.add(Dense(256, activation='relu'))
		model.add(Dropout(0.25))
		
		#Dense Block 3
		model.add(Dense(n_labels, activation='softmax'))

		return (model,model_name)		
		
		
		


def C3DP_4_LS_1_DE_2_V1(n_labels, input_shape):
		#image size (26,39) and seq length 32
		#val loss = 0.19396 on seven gestures
		#val loss = 0.26297 on nine gestures
		# Tunable parameters
		model_name = 'C3DP_4_LS_1_DE_2_V1'
		kernel_size = (3, 3, 3)
		strides = (1, 1, 1)
		extra_conv_blocks = 1

		model = Sequential()

		# 3D Conv Block 1 + Max Pooling
		model.add(Conv3D(32, kernel_size, strides=strides, activation='relu',
						 padding='same', input_shape=input_shape))
		model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

		# 3D Conv Block 2 + Max Pooling
		model.add(Conv3D(64, kernel_size, strides=strides, activation='relu',
						 padding='same'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

		# 3D Conv Block 3 + Max Pooling
		model.add(Conv3D(128, kernel_size, strides=strides, activation='relu',
						 padding='same'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

		# 3D Conv Block 4 + Max Pooling
		model.add(Conv3D(256, kernel_size, strides=strides, activation='relu',
						 padding='same'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

		# TD Flatten + Dropout
		model.add(TimeDistributed(Flatten()))
		model.add(Dropout(0.25))
		
		# LSTM Block 1 + Dropout
		model.add(LSTM(256, return_sequences=False, dropout=0.25))

		# Dense Block 1 + Dropout
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.25))
		
		# Dense Block 2
		model.add(Dense(n_labels, activation='softmax'))

		return (model,model_name)			

	
	



def C3DP_4_CD3_1_LS_2_DE_3(n_labels, input_shape):
		#image size (26,39) and seq length 32
		#val loss = 0.2512 on nine gestures
		# Tunable parameters
		model_name = 'C3DP_4_CD3_1_LS_2_DE_3'
		kernel_size = (3, 3, 3)
		strides = (1, 1, 1)
		extra_conv_blocks = 1

		model = Sequential()

		# 3D Conv Block 1 + Max Pooling
		model.add(Conv3D(32, kernel_size, strides=strides, activation='relu',
						 padding='same', input_shape=input_shape))
		model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

		# 3D Conv Block 2 + Max Pooling
		model.add(Conv3D(64, kernel_size, strides=strides, activation='relu',
						 padding='same'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

		# 3D Conv Block 3 + Max Pooling
		model.add(Conv3D(128, kernel_size, strides=strides, activation='relu',
						 padding='same'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
		
		# 3D Conv Block 4 + Max Pooling
		model.add(Conv3D(128, kernel_size, strides=strides, activation='relu',
						 padding='same'))
		model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
		
		# 3D Conv Block 5 + Dropout
		model.add(Conv3D(256, kernel_size, strides=strides, activation='relu',
						 padding='same'))
		model.add(Dropout(0.25))

		# TD Flatten + Dropout
		model.add(TimeDistributed(Flatten()))
		model.add(Dropout(0.25))
		
		# LSTM Block 1 + Dropout
		model.add(LSTM(256, return_sequences=True, dropout=0.25))
		
		# LSTM Block 2 + Dropout
		model.add(LSTM(128, return_sequences=False, dropout=0.25))

		# Dense Block 1 + Dropout
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.25))
		
		# Dense Block 2 + Dropout
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.25))
		
		# Dense Block 3
		model.add(Dense(n_labels, activation='softmax'))

		return (model,model_name)






def C3DP_3_LS_1_DE_2(n_labels, input_shape):
	#image size (26,39) and seq length 32
	#val loss = 0.27241
	# Tunable parameters
	model_name = 'C3DP_3_LS_1_DE_2'
	kernel_size = (3, 3, 3)
	strides = (1, 1, 1)
	extra_conv_blocks = 1

	model = Sequential()

	# 3D Conv Block 1 + Max Pooling
	model.add(Conv3D(32, (5,5,5), strides=strides, activation='relu',
					 padding='same', input_shape=input_shape))
	model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

	# 3D Conv Block 2 + Max Pooling
	model.add(Conv3D(64, kernel_size, strides=strides, activation='relu',
					 padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

	# 3D Conv Block 3 + Max Pooling
	model.add(Conv3D(128, kernel_size, strides=strides, activation='relu',
					 padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

	# TD Flatten + Dropout
	model.add(TimeDistributed(Flatten()))
	model.add(Dropout(0.25))
	
	# LSTM Block 1 + Dropout
	model.add(LSTM(256, return_sequences=False, dropout=0.25))

	# Dense Block 1 + Dropout
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.25))
	
	# Dense Block 2
	model.add(Dense(n_labels, activation='softmax'))

	return (model,model_name)








def C3DP_2_LS_2_DE_2(n_labels, input_shape):
	#image size (26,39) and seq length 32
	#val loss = 0.28893
	# Tunable parameters
	model_name = 'C3DP_2_LS_2_DE_2'
	kernel_size = (3, 3, 3)
	strides = (1, 1, 1)
	extra_conv_blocks = 1

	model = Sequential()

	# 3D Conv Block 1 + Max Pooling
	model.add(Conv3D(32, kernel_size, strides=strides, activation='relu',
					 padding='same', input_shape=input_shape))
	model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

	# 3D Conv Block 2 + Max Pooling
	model.add(Conv3D(64, kernel_size, strides=strides, activation='relu',
					 padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

	
	# TD Flatten + Dropout
	model.add(TimeDistributed(Flatten()))
	model.add(Dropout(0.25))
	
	# LSTM Block 1 + Dropout
	model.add(LSTM(256, return_sequences=True, dropout=0.25))
	
	# LSTM Block 2 + Dropout
	model.add(LSTM(256, return_sequences=False, dropout=0.25))
	
	# Dense Block 1 + Dropout
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.25))
	
	#Dense Block 2
	model.add(Dense(n_labels, activation='softmax'))

	return (model,model_name)		




def LRCN_C2DP_LS_2_DE_2(n_labels, input_shape):
	#image size (26,39) and seq length 32
	#val loss = 0.49394
	model_name = 'LRCN_C2DP_LS_2_DE_2'
	model = Sequential()
	
	# TD 2D Conv Block 1 + Max Pooling
	model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
	model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))
	
	# TD 2D Conv Block 2 + Max Pooling
	model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
	model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))

	# TD 2D Conv Block 3 + Max Pooling
	model.add(TimeDistributed(Conv2D(128, (4,4), activation='relu')))
	model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

	# TD 2D Conv Block 4 + Max Pooling
	model.add(TimeDistributed(Conv2D(256, (4,4), activation='relu')))
	model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

	# TD Dense Block 1 + Dropout
	model.add(TimeDistributed(Dense(128, activation='relu')))
	model.add(Dropout(0.25))
	
	# TD Flatten + Dropout
	model.add(TimeDistributed(Flatten()))
	model.add(Dropout(0.25))
	
	# LSTM Block 1 + Dropout
	model.add(LSTM(128, return_sequences=True, dropout=0.25))
	
	# LSTM Block 2 + Dropout
	model.add(LSTM(128, return_sequences=False, dropout=0.25))

	# Dense Block 2
	model.add(Dense(n_labels, activation='softmax'))

	return (model,model_name)











def C3DP_4_DE_3(n_labels, input_shape, k_size=(3,3,3)):
	#image size (26,39) and seq length 32
	model_name = 'C3DP_4_DE_3'
	model = Sequential()
	
	# 3D Conv Block 1 + Max Pooling
	model.add(Conv3D(filters=(64), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu'))
	model.add(MaxPooling3D(pool_size=(1,2,2), strides=(2,2,2)))
	
	# 3D Conv Block 2 + Max Pooling
	model.add(Conv3D(filters=(128), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu'))
	model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))
	
	# 3D Conv Block 3 + Max Pooling
	model.add(Conv3D(filters=(256), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu'))
	model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))

	# 3D Conv Block 4 + Max Pooling
	model.add(Conv3D(filters=(512), kernel_size=k_size, strides=(1,1,1), padding='same', activation='relu'))
	model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))

	model.add(Flatten())
	
	# Dense Block 1
	model.add(Dense(512, activation="relu"))
	
	# Dense Block 2
	model.add(Dense(512, activation="relu"))
	
	# Dense Block 3
	model.add(Dense(nb_classes, activation="softmax"))
	
	return (model,model_name)
	





def VGG16_LSTM(n_labels, input_shape):
	#image size (32,48) and seq length 32
	#val loss = 1.83813
	#use of vgg16 network for image analysis
	model_name = 'VGG16_LSTM'
	video = Input(shape=(32,32,48,3))
	cnn_base = VGG16(input_shape=(32,48,3),weights="imagenet",include_top=False)
	cnn_out = GlobalAveragePooling2D()(cnn_base.output)
	cnn = Model(input=cnn_base.input, output=cnn_out)
	cnn.trainable = False
	encoded_frames = TimeDistributed(cnn)(video)
	#lstm layer for time distribution
	encoded_sequence = LSTM(256)(encoded_frames)
	#2 dense blocks for result
	hidden_layer = Dense(output_dim=1024, activation="relu")(encoded_sequence)
	outputs = Dense(output_dim=n_labels, activation="softmax")(hidden_layer)
	model = Model([video], outputs)
	
	return (model,model_name)
	
	
def VGG16_GRU():
	#image size (224,224) and seq length 32
    #other way to use vgg16 for image analysis
	model_name = 'VGG16_GRU'
	pretrained_cnn = VGG16(weights='imagenet', include_top=False)
	for layer in pretrained_cnn.layers[:-5]:
		layer.trainable = False
    # input shape required by pretrained_cnn
	input = Input(shape = (224, 224, 3)) 
	x = pretrained_cnn(input)
	x = Flatten()(x)
	x = Dense(2048)(x)
	x = Dropout(0.5)(x)
	pretrained_cnn = Model(inputs = input, output = x)

	input_shape = (None, 224, 224, 3) # (seq_len, width, height, channel)
	model = Sequential()
	model.add(TimeDistributed(pretrained_cnn, input_shape=input_shape))
	model.add(GRU(1024, kernel_initializer='orthogonal', bias_initializer='ones', dropout=0.5, recurrent_dropout=0.5))
	model.add(Dense(categories, activation = 'softmax'))

	model.compile(loss='categorical_crossentropy',
				optimizer = optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1., clipvalue=0.5),
				metrics=['accuracy'])
	return (model,model_name)







	
	

def C2D_3_LS_1_DE_3(num_classes, input_shape):
	#image size (26,39) and seq length 32
	#val loss = 1.05066
	model_name = 'C2D_3_LS_1_DE_3'
	model=Sequential()
    
	# TD 2D Conv Block 1
	model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'), input_shape=input_shape))
	
	# TD 2D Conv Block 2
	model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu')))
	
	# TD 2D Conv Block 3 + Dropout
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	model.add(TimeDistributed(Dropout(0.25)))
    
	# TD Flatten
	model.add(TimeDistributed(Flatten()))
	
	# TD Dense Block 1
	model.add(TimeDistributed(Dense(512, activation='relu')))
    
	# TD Dense Block 2
	model.add(TimeDistributed(Dense(32, activation='relu')))
    
	# LSTM Block 1
	model.add(LSTM(32, return_sequences=True))

	# TD Dense Block 3 + 1D Average Pooling
	model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
	model.add(GlobalAveragePooling1D())

	return (model,model_name)


def C2D_2_LS_1_DE_3(num_classes, input_shape):
	#image size (26,39) and seq length 32
	model_name = 'C2D_2_LS_1_DE_3'
	model=Sequential()
	
	# TD 2D Conv Block 1
	model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'), input_shape=input_shape))
	
	# TD 2D Conv Block 2 + Max Pooling + Dropout
	model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
	model.add(TimeDistributed(Dropout(0.25)))
    
	# TD Flatten
	model.add(TimeDistributed(Flatten()))
	
	# TD Dense Block 1
	model.add(TimeDistributed(Dense(512, activation='relu')))
    
	# TD Dense Block 2
	model.add(TimeDistributed(Dense(32, activation='relu')))
    
	# LSTM Block 1
	model.add(LSTM(32, return_sequences=True))

	# TD Dense Block 3 + 1D Average Pooling
	model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
	model.add(GlobalAveragePooling1D())

	return (model,model_name)




def BUILD_MODEL(n_labels, input_shape):
	#image size (26,39) and seq length 32
	#try to combine between two models
	model_name = 'BUILD_MODEL'
	rgb_model = C2D_3_LS_1_DE_3(n_labels, input_shape)
	flow_model = C2D_2_LS_1_DE_3(n_labels, input_shape)

	model = Sequential()
	#model.add(Merge([rgb_model, flow_model], mode='ave'))
	#model.add(concatenate([rgb_model, flow_model]))
	model.add(average([rgb_model, flow_model]))
	#model.add(average([rgb_model, flow_model]))

	model.compile(loss='categorical_crossentropy',
				  optimizer='rmsprop',
				  metrics=['accuracy'])

	return (model,model_name)
		
		
		
		
		
		
		
		
def INCEPTIONV3_LSTM(n_labels, input_shape):
	#image size (76,114) and seq length 32
	#val loss = 1.85308
	model_name = 'INCEPTIONV3_LSTM'
	#use inception v3 network for image analysis
	video = Input(shape=(32,76,114,3),name='video_input')
	cnn = InceptionV3(
		input_shape=(76,114,3),
		weights='imagenet',
		include_top=False,
		pooling='avg')
	cnn.trainable = False
	# wrap cnn into Lambda and pass it into TimeDistributed
	encoded_frame = TimeDistributed(cnn)(video)
	#use lstm for time distribution
	encoded_vid = LSTM(256)(encoded_frame)
	#use dense for result
	hidden_layer = Dense(128, activation='relu')(encoded_vid)
	outputs = Dense(output_dim=n_labels, activation="softmax")(hidden_layer)
	model = Model([video], outputs)

	
	return (model,model_name)		
	
	
	




def get_model(n_labels, input_shape, weights='imagenet'):
    # create the base pre-trained model
    base_model = InceptionV3(weights=weights, input_shape=(76,114,3), include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(n_labels, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def freeze_all_but_mid_and_top(n_labels, input_shape):
	"""After we fine-tune the dense layers, train deeper."""
	# we chose to train the top 2 inception blocks, i.e. we will freeze
	# the first 172 layers and unfreeze the rest:
	model_name = 'INCEPTIONV3_MODIFIED'
	model = get_model(n_labels, input_shape)
	for layer in model.layers[:172]:
		layer.trainable = False
	for layer in model.layers[172:]:
		layer.trainable = True

	return model,model_name




def lstm(n_labels, input_shape):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model_name = 'LSTM'
        model = Sequential()
        model.add(LSTM(2048, return_sequences=False,
                       input_shape=(32,2048),
                       dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_labels, activation='softmax'))

        return model,model_name

def lrcn(n_labels, input_shape):
    """Build a CNN into RNN.
    Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
            steering-models/community-models/chauffeur/models.py
    Heavily influenced by VGG-16:
        https://arxiv.org/abs/1409.1556
    Also known as an LRCN:
        https://arxiv.org/pdf/1411.4389.pdf
    """
    model_name = 'lrcn'
    model = Sequential()

    model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
        activation='relu', padding='same'), input_shape=input_shape))
    model.add(TimeDistributed(Conv2D(32, (3,3),
        kernel_initializer="he_normal", activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(64, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(64, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(128, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(128, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(256, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(256, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
    
    model.add(TimeDistributed(Conv2D(512, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(512, (3,3),
        padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    model.add(Dense(n_labels, activation='softmax'))

    return model,model_name
		

