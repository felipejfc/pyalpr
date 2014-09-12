import cv2
import numpy as np

class Char:
	def __init__(self, img, x):
		self.img = img
		self.x = x


#def projectHistogram(img, orientation):
#	h,w = img.shape[:2]
#	sz = h if orientation else w
#	mHist = np.zeros((1, sz), np.float32)
#	for i in range(0, sz):
#		data = img[:][i] if orientation else img[i][:]
#		mHist[i] = cv2.countNonZero(data)
#
#	minVal,maxVal,minLoc, maxLoc = cv2.minMaxLoc(mhist)
#	if maxVal > 0:
#		mHist *= 1.0/maxVal
#
#	return mHist
#
#def features(in, sizeData):
#	VERTICAL  = 0
#	HORIZONAL = 1
#	vHist = projectHistogram(in, VERTICAL)
#	hHist = projectHistogram(in, HORIZONAL)
#	hV,wV = vHist.shape[:2]
#	hH,wH = hHist.shape[:2]
#	lowData = cv2.resize(in, (sizeData))
#	hL,wL = lowData.shape[:2]
#	numCols=wV+wH+wL*wL;
#	out = np.zeros((1, numCols), np.float32)
#	j = 0
#	for i in range (0, wV):
#		out[j] = vHist[i]
#		j+=1
#	for i in range(0, wH):
#		out[j] = hHist[i]
#		j+=1
#	for i in range(0, wL):




#Aproximated aspect for characters on the plate will be 1.0, we will use 35 percent error margin
def verifySizes(contour):
	aspect = 0.8
	error = 0.35

	x,y,w,h = cv2.boundingRect(contour)
	charAspect = float(w)/h

	minHeight = 12
	maxHeight = 27
	#aspect para o numero 1 eh aprox. 0.2
	minAspect = 0.2
	maxAspect = aspect+aspect*error
	area = cv2.contourArea(contour)
	#print "Area: " + str(area) + " Char aspect: " + str(charAspect) + " Char Height: "+ str(h) 
	return charAspect < maxAspect and charAspect > minAspect and h < maxHeight and h > minHeight

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
	#cv2.imshow('cha '+str(charN), warpImage)
	return warpImage

def segment(plate):
	charN = 0
	output = []
	#Threshold + findContours + iterar neles para deletar invalidos
	plateThreshold = cv2.threshold(plate, 60, 255, cv2.THRESH_BINARY_INV)
	plateContours  = plateThreshold[1].copy();
	contours = cv2.findContours(plateContours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	result = plateThreshold[1].copy()
	result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB);
	cv2.drawContours(result, contours[0], -1, (255,0,0),1)
	cv2.imshow('THRESH_BINARY_INV + findContours', result)
	for contour in contours[0]:
		mr = cv2.boundingRect(contour)
		cv2.rectangle(result, (mr[0],mr[1]),(mr[0]+mr[2],mr[1]+mr[3]), (0,255,0))
		if(verifySizes(contour)):
			crop = plate[mr[1]:mr[1]+mr[3],mr[0]:mr[0]+mr[2]]
			processedChar = preprocess(crop)
			output.append(Char(processedChar,mr[0]))
			cv2.rectangle(result, (mr[0],mr[1]),(mr[0]+mr[2],mr[1]+mr[3]), (0,125,255));
	cv2.imshow('result', result)
	return output
