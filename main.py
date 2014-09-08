import cv2
import random

#Dimencoes de acordo com o detran, 40x13(cm), aspecto = 40/13 = 3,07
def verifySize(contour):
   error = 0.4
   aspect = 3.0769
   min = 15*aspect*15
   max = 125*aspect*125
   rmin = aspect-aspect*error
   rmax = aspect+aspect*error

   #print contour

   area = cv2.contourArea(contour)
   #print(area)

   x,y,w,h = cv2.boundingRect(contour)
   r = float(w)/h
   
   if r < 1:
      r = float(h)/w

   #print('aspect: '+ str(r)+ ', area: '+ str(area))

   if (area < min or area > max) or (r < rmin or r > rmax):
      return False
   return True

def main():
   img = cv2.imread('resources/plates/placa9.jpg')

   gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   gray = cv2.blur(gray,(5,5))

   img_sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3, 1, 1, cv2.BORDER_DEFAULT)

   img_threshold = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

   element = cv2.getStructuringElement(cv2.MORPH_RECT,(17, 3));
   img_threshold = cv2.morphologyEx(img_threshold[1], cv2.MORPH_CLOSE, element);


   #Encontrar contornos

   #Python: cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy
   contours = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   contours = [contour for contour in contours[0] if verifySize(contour)]
   rects = contours

   #Contorna os contornos encontrados de azul
   result = img
   cv2.drawContours(result, contours, -1, (255,0,0),1)

   for rect in rects:
      (center_x,center_y),radius = cv2.minEnclosingCircle(rect)
      cv2.circle(result, (int(center_x),int(center_y)), 3, (0,255,0), -1)
      x,y,w,h = cv2.boundingRect(rect)
      minSize = w if w < h else h;
      minSize = minSize-minSize*0.5;

      loDiff = 30;
      upDiff = 30;
      connectivity = 4;
      newMaskVal = 255;
      numSeeds = 10;
      flags = connectivity + (newMaskVal << 8 ) + cv2.FLOODFILL_FIXED_RANGE + cv2.FLOODFILL_MASK_ONLY;

      for i in range(0,numSeeds):
         seed   = {}
         seed['x'] = center_x+(random.randrange(1,1000) % int(minSize))-(minSize/2);
         seed['y'] = center_y+(random.randrange(1,1000) % int(minSize))-(minSize/2);
         cv2.circle(result, (int(seed['x']),int(seed['y'])), 1, (255,255,0), -1);

   cv2.imshow('image',result)
   cv2.waitKey(0)
   cv2.destroyAllWindows()


if __name__ == "__main__":
    main()