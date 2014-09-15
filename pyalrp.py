import cv2
import numpy as np
import random
import sys
import plate_extraction as pe
import ocr
import argparse
import json
import os.path

parser = argparse.ArgumentParser(description='ALPR (Automatic License Plate Recognition made in Python)')
parser.add_argument('-I', '--image', help='image for ALPR')
parser.add_argument('--generate', action='store_true', help='generate the OCR neural network train data based on the images in train folder',
					default = False)
parser.add_argument('--show-steps', action='store_true', help='show the steps to segment the input image',
					default = False)
parser.add_argument('--save-output', action='store_true', help='save the output results',
					default = False)


args = parser.parse_args()

def main():
	DEBUG = True if args.show_steps else False 
	SAVE = True if args.save_output else False
	if(args.generate):
		ocr.generateTrainData()
	else:
		if not os.path.isfile("OCR.json"):
			print "Training file OCR.json not found, first run the script with --generate flag"
			return 1
		img = cv2.imread(args.image)
		if(DEBUG):
			cv2.imshow('original',img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		plates = pe.extractPlates(img,DEBUG)
		json_data=open('OCR.json')
		trainDataJSON = json.load(json_data)
		trainData = trainDataJSON["trainDataF15"]
		trainLabels = trainDataJSON["labels"]
		ann = ocr.train(trainData, trainLabels, 91, 36)
		if len(plates) > 0:
			plate_id = 0
			for plate in plates:
				plate_id+=1
				chars = ocr.segment(plate,DEBUG)
				chars = sorted(chars,key=lambda char: char.x)
				res = ""
				for char in chars:
					res = res+ocr.classify(ocr.features(char.img,(15,15)), ann)
				print(res[0:3]+"-"+res[3:7])
			if DEBUG:
				cv2.waitKey(0)
				cv2.destroyAllWindows()

		else:
			print('No plates found :/')


if __name__ == "__main__":
   main()