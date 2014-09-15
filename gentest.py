import Image
import ImageDraw
import ImageFont
import cv2
import numpy as np

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

image = Image.new("RGBA", (1190,280), (255,255,255))
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("resources/MANDATORY.ttf", 80)

draw.text((10, 10), "0 1 2 3 4 5 6 7 8 9 A B C D E F", (0,0,0), font=font)
draw.text((10, 90), "G H I J K L M N O P Q R S T U V", (0,0,0), font=font)
draw.text((10, 170), "W X Y Z", (0,0,0), font=font)
image.save("resources/font.png")

charN = 0
output = []
img = cv2.imread('resources/font.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
img = cv2.resize(img,(460,100))
img = cv2.blur(img,(2,2))
img = cv2.equalizeHist(img)
threshold = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)
#if DEBUG:
#cv2.imshow('t',threshold[1])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

imgContours  = threshold[1].copy();
contours = cv2.findContours(imgContours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
result = threshold[1].copy()
result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB);
#if DEBUG:
#cv2.drawContours(result, contours[0], -1, (255,0,0),1)
#cv2.imshow('THRESH_BINARY_INV + findContours', result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

aux = 0

for contour in contours[0]:
	mr = cv2.boundingRect(contour)
	cv2.rectangle(result, (mr[0],mr[1]),(mr[0]+mr[2],mr[1]+mr[3]), (0,255,0))
	crop = img[mr[1]:mr[1]+mr[3],mr[0]:mr[0]+mr[2]]
	processedChar = preprocess(crop)
	processedChar = cv2.resize(processedChar,(15,15))
	cv2.imwrite(str(aux)+".jpg",processedChar)
	aux+=1
#	cv2.imshow('a', processedChar)
	cv2.rectangle(result, (mr[0],mr[1]),(mr[0]+mr[2],mr[1]+mr[3]), (0,125,255));
#if DEBUG:
#cv2.imshow('Rects', result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#img_resized = image.resize((10,10), Image.ANTIALIAS)
