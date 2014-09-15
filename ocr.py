import cv2
import numpy as np
import json
from cv2 import ANN_MLP

class Char:
	def __init__(self, img, x):
		self.img = img
		self.x = x

chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 
 		 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
		 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

#Project the histogram of the input img
def projectHistogram(img, orientation):
	h,w = img.shape[:2]
	sz = h if orientation==1 else w
	mHist = np.zeros((1, sz), np.float32)
	for i in range(0, sz):
		data = img[:][i] if orientation else img[i][:]
		mHist[0][i] = cv2.countNonZero(data)

	minVal,maxVal,minLoc, maxLoc = cv2.minMaxLoc(mHist)
	if maxVal > 0:
		mHist *= 1.0/maxVal
	return mHist

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#Train a MPL neural network to recognize chars
def train(trainData, classes, nLayers, nOutLayers):
	layers = np.array([len(trainData[0]), nLayers, nOutLayers])
	ann = ANN_MLP(layers, cv2.ANN_MLP_SIGMOID_SYM,1,1)

	inputs = np.empty((len(trainData), len(trainData[0])), 'float' )

	for i in range(len(trainData)):
	    a = np.array(list(trainData[i]))
	    inputs[i,:] = a[:]

	#outputs
	outputs = np.zeros((len(trainData),36))
	for i in range(len(trainData)):
		outputs[i][classes[i]] = 1

	# Some parameters for learning.  Step size is the gradient step size
	# for backpropogation.
	step_size = 0.01
	# Momentum can be ignored for this example.
	momentum = 0.0
	# Max steps of training
	nsteps = 10000
	# Error threshold for halting training
	max_err = 0.0001
	# When to stop: whichever comes first, count or error
	condition = cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS
	# Tuple of termination criteria: first condition, then # steps, then
	# error tolerance second and third things are ignored if not implied
	# by condition
	criteria = (condition, nsteps, max_err)

	# params is a dictionary with relevant things for NNet training.
	params = dict( term_crit = criteria, 
	               train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP, 
	               bp_dw_scale = step_size, 
	               bp_moment_scale = momentum )

	# Train our network
	num_iter = ann.train(inputs, outputs, None, params=params)


	# Create a matrix of predictions
#	predictions = np.empty_like(outputs)

	# See how the network did.
#	ann.predict(inputs, predictions)

	# Compute sum of squared errors
#	sse = np.sum( (outputs - predictions)**2 )

	# Compute # correct
#	true_labels = np.argmax( outputs, axis=0 )
#	pred_labels = np.argmax( predictions, axis=0 )
#	num_correct = np.sum( true_labels == pred_labels )

#	print 'predictions:'
#	print predictions
#	print 'sum sq. err:', sse
#	print 'accuracy:', float(num_correct)/len(true_labels)

	return ann

#Classify the input char
def classify(features, ann):
	a = np.array(list(features))
	inputs = np.empty((1, len(features)), 'float' )
	inputs[0,:] = a[:]

	predictions = np.zeros((1,36))
	ann.predict(inputs, predictions)
	minV, maxV, minL, maxL = cv2.minMaxLoc(predictions);
	return chars[maxL[0]]

#Generate the OCR.json train data file
def generateTrainData():
	#Chars that we wish to train
	numTrainChars = [6 , 8 , 9 , 9 , 6 , 9 , 8 , 8 , 6 , 9 , 4 , 4 , 5 , 
					 3 , 3 , 7 , 5 , 5 , 3 , 5 , 7 , 4 , 5 , 6 , 6 , 5 , 
					 6 , 3 , 5 , 4 , 5 , 3 , 3 , 3 , 4 , 3]
	trainDataF10 =   []
	trainDataF15 =   []
	trainDataF20 =   []
	trainingLabels = []
	for i in range(0, len(chars)):
		for j in range(1, numTrainChars[i]+1):
			img = cv2.imread("resources/samples/"+chars[i]+"_"+str(j)+".jpg",0)
			f10 = features(img,(10,10))
			f15 = features(img,(15,15))
			f20 = features(img,(20,20))

			trainDataF10.append(f10)
			trainDataF15.append(f15)
			trainDataF20.append(f20)
			trainingLabels.append(i)
	with open('OCR.json', 'w') as outfile:
		json.dump({'trainDataF10':trainDataF10, 'trainDataF15':trainDataF15, 'trainDataF20': trainDataF20, 'labels':trainingLabels}, outfile,cls=NumpyAwareJSONEncoder)

#Create an array containing the projection of vertical and horizontal histogram of the char image and its binary data
def features(img, sizeData):
	VERTICAL  = 0
	HORIZONAL = 1
	lowData = cv2.resize(img, sizeData)
	vHist = projectHistogram(lowData, VERTICAL)
	hHist = projectHistogram(lowData, HORIZONAL)
	hV,wV = vHist.shape[:2]
	hH,wH = hHist.shape[:2]
	#print(lowData)
	hL,wL = lowData.shape[:2]
	numCols=wV+wH+wL*wL;
	out = np.zeros((numCols), 'float')
	j = 0
	for i in range (0, wV):
		out[j] = vHist[0][i]
		j+=1
	for i in range(0, wH):
		out[j] = hHist[0][i]
		j+=1
	for x in range(0, hL):
		for y in range(0, wL):
			out[j] = float(lowData[x][y])
			j+=1
	#print "out" + str(out)
	return out

#Aproximated aspect for characters on the plate will be 1.0, we will use 35 percent error margin
def verifySizes(contour):
	aspect = 0.8
	error = 0.35

	x,y,w,h = cv2.boundingRect(contour)
	charAspect = float(w)/h

	minHeight = 12
	maxHeight = 27
	#aspect for number 1 is aprox. 0.2
	minAspect = 0.2
	maxAspect = aspect+aspect*error
	area = cv2.contourArea(contour)
	#print "Area: " + str(area) + " Char aspect: " + str(charAspect) + " Char Height: "+ str(h) 
	return charAspect < maxAspect and charAspect > minAspect and h < maxHeight and h > minHeight

#Preprocess the char to normalize the test data
def preprocess(char):
	a,char = cv2.threshold(char, 60, 255, cv2.THRESH_BINARY_INV)
	h,w = char.shape[:2]
	transf = np.identity(3)
	transf = transf[0:2,0:3]
	m = max(h,w)
	transf[0][2] = m/2.0 - w/2.0
	transf[1][2] = m/2.0 - h/2.0
	warpImage = np.zeros((m,m),np.uint8)
	warpImage=cv2.warpAffine(char, transf, (m,m), warpImage, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (0) );
	return warpImage

#Find the chars in the plate image
def segment(plate, DEBUG):
	charN = 0
	output = []
	#Threshold + findContours + iterar to delete invalid
	plateThreshold = cv2.threshold(plate, 60, 255, cv2.THRESH_BINARY_INV)
	plateContours  = plateThreshold[1].copy();
	contours = cv2.findContours(plateContours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	result = plateThreshold[1].copy()
	result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB);
	cv2.drawContours(result, contours[0], -1, (255,0,0),1)
	#cv2.imshow('THRESH_BINARY_INV + findContours', result)
	for contour in contours[0]:
		mr = cv2.boundingRect(contour)
		cv2.rectangle(result, (mr[0],mr[1]),(mr[0]+mr[2],mr[1]+mr[3]), (0,255,0))
		if(verifySizes(contour)):
			crop = plate[mr[1]:mr[1]+mr[3],mr[0]:mr[0]+mr[2]]
			processedChar = preprocess(crop)
			output.append(Char(processedChar,mr[0]))
			cv2.rectangle(result, (mr[0],mr[1]),(mr[0]+mr[2],mr[1]+mr[3]), (0,125,255));
	if DEBUG:
		cv2.imshow('detected chars', result)

	return output
