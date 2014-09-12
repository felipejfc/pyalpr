import cv2
import numpy as np
import random
import sys
import plate_extraction as pe
import ocr

def main():
	img = cv2.imread(sys.argv[1])
	plates = pe.extractPlates(img)
	if len(plates) > 0:
		plate_id = 0
		for plate in plates:
			plate_id+=1
			cv2.imshow('plate '+str(plate_id), plate)
			chars = ocr.segment(plate)
			for char in chars:
				cv2.imshow('x: '+str(char.x),char.img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		print('No plates found :/')


if __name__ == "__main__":
   main()