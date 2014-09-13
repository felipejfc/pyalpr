import cv2
import numpy as np
import random
import sys
import plate_extraction as pe
import ocr
import argparse

parser = argparse.ArgumentParser(description='ALPR (Automatic License Plate Recognition made in Python)')
parser.add_argument('-I', '--image', help='image for ALPR')
parser.add_argument('--train', action='store_true', help='train the OCR neural network with the images in the folder train',
					default = False)

args = parser.parse_args()

def main():
	if(args.train):
		ocr.train()
	else:
		img = cv2.imread(args.image)
		plates = pe.extractPlates(img)
		if len(plates) > 0:
			plate_id = 0
			for plate in plates:
				plate_id+=1
				cv2.imshow('plate '+str(plate_id), plate)
				chars = ocr.segment(plate)
				for char in chars:
					cv2.imshow('x: '+str(char.x),char.img)
					ocr.features(char.img,(10,10))
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		else:
			print('No plates found :/')


if __name__ == "__main__":
   main()