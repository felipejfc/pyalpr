import cv2
import numpy as np
import random
import sys
import plate_extraction as pe

def main():
   img = cv2.imread(sys.argv[1])
   plates = pe.extractPlates(img)
   plate_id = 0
   for plate in plates:
      plate_id+=1
      cv2.imshow('plate '+str(plate_id), plate)
   cv2.waitKey(0)
   cv2.destroyAllWindows()


if __name__ == "__main__":
   main()