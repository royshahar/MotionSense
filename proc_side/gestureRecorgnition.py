import json
import numpy as np
import cv2
import tensorflow as tf
import win32gui
import win32process
import os
import pyautogui
import psutil
from SysTrayIcon import SysTrayIcon
from threading import Thread

#MODEL_PATH = "model\\model.hdf5"		
MODEL_PATH = "model\\3DCNN_LSTM_RGB.h5"		
IMAGE_SIZE = (39,26)
#IMAGE_SIZE = (48,32)
SEQUENCE_LENGTH = 10
CURRENT_PATH = ""
#CATEGORIES = ['Swiping Left', 'Swiping Right', 'Swiping Down', 'Swiping Up', 'No gesture', 'Doing other things']
#CATEGORIES = ['Swiping Left', 'Swiping Right', 'Swiping Down', 'Swiping Up', 'Stop Sign', 'No gesture', 'Doing other things']
#CATEGORIES = ['Swiping Left', 'Swiping Right', 'Swiping Down', 'Swiping Up', 'Thumb Up', 'Thumb Down', 'Stop Sign', 'No gesture', 'Doing other things']
CATEGORIES = ['Swiping Left', 'Swiping Right', 'Thumb Down', 'Thumb Up', 'Drumming Fingers','Stop Sign', 'Zooming In With Two Fingers', 'Zooming Out With Two Fingers', 'No gesture', 'Doing other things']
show_cam = False
show_preds = False

#this class analyze the frames taken from the camera and make predictions based on them.
class ImageAnalysis:
	image_size = IMAGE_SIZE
	number_of_frames_in_sequence = SEQUENCE_LENGTH
	
	"""init function, initialize the two streams and load the model"""
	def __init__(self):
		self.model = tf.keras.models.load_model(MODEL_PATH)
		#self.model = load_model(MODEL_PATH)
		self.predict_stream = {"index": -1, "stream": 0}
		self.curr_stream = []
		print('init ImageAnalysis \n')

	"""The function fixes img to fit model"""
	def prepreocess_img(self,  img_array):
		return (img_array / 255. )
	
	"""The function makes prediction on the fream stream using the model"""
	def predict(self):		
		x = []		 
		x.append(self.curr_stream)
		X = np.array(x)
		prediction = self.model.predict(X)
		
		if show_preds:
			self.show_pred(prediction[0])
		
		index = np.argmax(prediction[0])
		accuracy = np.max(prediction[0])
		if CATEGORIES[index] != "No gesture" and CATEGORIES[index] != "Doing other things" and accuracy > 0.7:
			if self.predict_stream["index"] == index:
				if self.predict_stream["stream"] != 5:
					self.predict_stream["stream"] += 1
				else:
					self.predict_stream["stream"] += 1
					self.curr_stream = []
					return CATEGORIES[index]
					
			else:
				self.predict_stream["index"] = index
				self.predict_stream["stream"] = 1
				
		elif accuracy > 0.9: # Detected "No gesture" or "Doing other things" with high accuracy		
			self.predict_stream["index"] = -1
			self.predict_stream["stream"] = 0
		
		return ""
	
	"""The function returns the length of the frame stream"""
	def img_num(self):
		return len( self.curr_stream)

	"""The function takes frame, fix it to fit the model and push him into the frame stream"""
	def push_img (self, frame):
		frame = cv2.resize(frame, self.image_size)
		img_arr = np.array(frame)
		
		#img_arr = img_arr.reshape((self.image_size[1], self.image_size[0], 1))
		
		img = self.prepreocess_img(img_arr)
		self.curr_stream.append(img)
		if len( self.curr_stream ) > self.number_of_frames_in_sequence:
			self.curr_stream.pop(0)
	
	"""The function shows the predictions perecantage for each gesture in a screen"""
	def show_pred(self, preds):
		preds = list(preds)
		canvas = np.zeros((250, 300, 3), dtype="uint8")
		for (i, (label, prob)) in enumerate(zip(CATEGORIES, preds)):
			# construct the label text
			text = "{}: {:.2f}%".format(label, prob * 100)

			w = int(prob * 300)
			cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
			cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
		cv2.imshow("Probabilities", canvas)			


#this class is responsiable for taking frames from the camera
class CameraWrapper:
	isCurrentlyPreview = False
	
	"""init function, initialize imageAnalysis class"""
	def __init__(self):
		print('init CameraWrapper\n')
		self.image_analysis = ImageAnalysis()
	
	"""the function takes frame from the camera"""
	def CaptureFrame(self,cap):
		# Capture frame-by-frame
		ret, frame = cap.read()
		#frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
		return frame
		
		
#this class use the prediction and the predefined defintions to do actions using the operating system
class SystemController:
	
	"""init function, initialize the defintion database"""
	def __init__(self):
		self.config_DB = self.read_from_DB()
		print('init SystemController\n')
		
	"""The function reads from defintions.json and returns a dictionary of user defintions for the software"""
	def read_from_DB(self):
		path = CURRENT_PATH + "\\definitions.json"
		with open(path,'r') as file:
			dict_str = file.read()
			return json.loads(dict_str)
	
	
	"""The function communicate with the operating system to make an action based on the defintion database and the prediction"""
	def do_action(self,action):
		print("do action\n")
		program_name = "general"
		
		# Get active window
		try:
			pid = win32process.GetWindowThreadProcessId(win32gui.GetForegroundWindow()) #This produces a list of PIDs active window relates to
			program_name = psutil.Process(pid[-1]).exe()
		except Exception as e:
			print(str(e))	
			
			
		#print("Active Program: " + program_name + "\n")
		operation = ""
		key_in_dict = "" 
		for key in self.config_DB.keys():
			if key.split("~")[1] == program_name:
				key_in_dict = key
				break
			elif key.split("~")[0] == "Default":
				key_in_dict = key
			
				
		if key_in_dict != "": # Found action for the program
			if action in self.config_DB[key_in_dict].keys():
				operation = self.config_DB[key_in_dict][action]
				#operation = operation.lower()
			print("Key to press: " + str(operation) + "\n")
			
			try: #press the key
				if len(operation.split("+")) > 1 and len(operation) > 1:
					if len(operation.split("+")) == 2:
						pyautogui.hotkey(operation.split("+")[0], operation.split("+")[1])
					else:
						pyautogui.hotkey(operation.split("+")[0], operation.split("+")[1], operation.split("+")[2])
				else:
					if not (operation.isalpha() and len(operation) == 1):
						operation = operation.lower()
					pyautogui.press(operation) 
					
				print("pressed\n")
			except Exception as e:
				print(str(e))		
		else:
			print("No definition has been setted\n")
			
		
"""The function start the operation of the process and keeps it running until the user stops it"""
def startPreview(camera_wrapper, system_controller):
	print('in start preview\n')
	number_of_frames_in_sequence = SEQUENCE_LENGTH
	cap = cv2.VideoCapture(0)
	camera_wrapper.isCurrentlyPreview = True
	even = True
	while True:
		if not cap.isOpened():
			print(("Couldn't find webcam\n"))
			break
		
		frame = camera_wrapper.CaptureFrame(cap) # Capture a frame from camera
		if show_cam:
			cv2.imshow('frame',frame)	
		if not even: # Predict each second image
			even = True
			continue
		else:
			even = False
		
		if camera_wrapper.image_analysis.img_num() < number_of_frames_in_sequence:
			camera_wrapper.image_analysis.push_img(frame)
		else:			
			action = camera_wrapper.image_analysis.predict()
			if action != "": # Found key to press
				print("\n" + action + "\n")
				system_controller.do_action(action)
			camera_wrapper.image_analysis.push_img(frame)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
	camera_wrapper.isCurrentlyPreview = False

"""The function launches the gui from the tray icon"""
def launch(sysTrayIcon):
	subprocess.Popen("runGui.bat")

"""The function used as sysTrayIcon destroy function"""	
def bye(sysTrayIcon): pass		
	
"""main function, responsiable for calling startPreview function"""
def main():
	print("Start")
	global CURRENT_PATH
	global show_cam
	global show_preds
	CURRENT_PATH = os.path.dirname(os.path.abspath(__file__)) # Get the file's path
	
	try:
		path = CURRENT_PATH + "\\settings.txt"
		with open(path,'r') as file:
			data = file.read()
			show_cam = bool(int(data.split(",")[0][-1]))
			show_preds = bool(int(data.split(",")[1][-1]))
	except Exception as e:
		print("couldn't load settings")
		
		
	icon = "icon.ico"
	hover_text = "MotionSense"
	menu_options = (('Launch MotionSense', None, launch),)
	thread = Thread(target = SysTrayIcon, args = (icon, hover_text, menu_options, bye, 1, ))
	thread.start()	
	
	
	camera_wrapper = CameraWrapper()	
	system_controller = SystemController()
	startPreview(camera_wrapper, system_controller)
	
	
if __name__ == '__main__':
	main()