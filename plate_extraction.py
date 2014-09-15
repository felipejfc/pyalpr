import cv2
import numpy as np
import random

#Dimencoes de acordo com o detran, 40x13(cm), aspecto = 40/13 = 3,07, por algum motivo os resultados sao melhores quando um resultado superior eh usado
def verifySize(contour):
	error = 0.4
	aspect = 4.0769
	min = 15*aspect*15
	max = 125*aspect*125
	rmin = aspect-aspect*error
	rmax = aspect+aspect*error

	area = cv2.contourArea(contour)

	x,y,w,h = cv2.boundingRect(contour)
	r = float(w)/h

	if (area < min or area > max) or (r < rmin or r > rmax):
		return False
	return True

def verifyMaskAspect(w, h):
	error = 0.4
	aspect = 4.0769
	min = 15*aspect*15
	max = 125*aspect*125
	rmin = aspect-aspect*error
	rmax = aspect+aspect*error

	area = w*h

	r = float(w)/h

	if(r < 1):
		r = h/w

	if (area < min or area > max) or (r < rmin or r > rmax):
		return False
	return True   

def extractPlates(img,DEBUG):
	results = []
	#cv2.imshow('original', img)
	imgSize = img.shape[:2]
	#Preprocessamento (blur + sobel + threshold + morphologyEx)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = cv2.blur(gray,(5,5))
	img_sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3, 1, 1, cv2.BORDER_DEFAULT)
	#cv2.imshow("sobel",img_sobel)
	img_threshold = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
	element = cv2.getStructuringElement(cv2.MORPH_RECT,(17, 3));
	#cv2.imshow('threshold',img_threshold[1])
	img_threshold = cv2.morphologyEx(img_threshold[1], cv2.MORPH_CLOSE, element);
	#cv2.imshow('threshold+morphologyEx',img_threshold)

	#Segmentacao (encontrar contornos, verificar se sao validos, rodar floodfill em pontos aleatorios ao redor do centro de massa do contorno)
	contours = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contours = [contour for contour in contours[0] if verifySize(contour)]
	rects = contours
	result = img.copy()
	cv2.drawContours(result, contours, -1, (255,0,0),1)

#	cv2.imshow('result',result)

	#floodfill
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
		flags = connectivity | (newMaskVal << 8 ) + cv2.FLOODFILL_FIXED_RANGE + cv2.FLOODFILL_MASK_ONLY;
		mask = np.zeros((imgSize[0]+2, imgSize[1]+2), np.uint8)

		for i in range(0,numSeeds):
			seed   = {}
			seed['x'] = center_x+(random.randrange(-10000,10000) % int(w/2));
			seed['y'] = center_y+(random.randrange(-10000,10000) % int(h/2.5));
			cv2.circle(result, (int(seed['x']),int(seed['y'])), 1, (255,255,0), -1);
			area,fill_rect = cv2.floodFill(img, mask, (int(seed['x']),int(seed['y'])), (255,0,0), (loDiff, loDiff, loDiff), (upDiff, upDiff, upDiff), flags);

		#cv2.imshow('result',result)

		#Pega os pontos brancos encontrados na mascara para depois fazer o minAreaRect que contem esses pontos
		pointsOfInterest = []
		for i in range(0,len(mask)):
			for j in range (0,len(mask[i])):
				if mask[i][j] == 255:
					pointsOfInterest.append([j,i])

		#Faz um minAreaRect desses pontos
		npArrayPoints = np.array(pointsOfInterest)
		minRect = cv2.minAreaRect(npArrayPoints)
		box = cv2.cv.BoxPoints(minRect)
		box = np.int0(box)
		cv2.drawContours(mask,[box],0,(255,0,0),2)

		maskW, maskH = minRect[1][0], minRect[1][1]
		maskRotation = minRect[2]
		maskCenter = minRect[0]
		if(DEBUG):
			cv2.imshow('candidate regions',result)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		#cv2.imshow('result',mask)

		#Verifica se o tamanho dessa mascara eh compativel com o aspecto de uma placa de transito
		#Com a segmentacao acabada, fazer crop da regiao de imagem, redimensionar ela e equalizar a luz
		if verifyMaskAspect(maskW,maskH):
			#Fix para comportamento estranho que acontece com o ratio!
			if(maskW / maskH < 1):
				maskW, maskH = maskH, maskW
			imgCrop = cv2.getRectSubPix(img, (int(maskW),int(maskH)), maskCenter)
			#Redimensionar imagem cropada para tamanho padrao e aplicar histograma de equalizacao de luz
			resultResized = cv2.resize(imgCrop,(144,33))
			#Equalizar imagem cropada (gray + blur + equalizeHist)
			grayResult = cv2.cvtColor(resultResized,cv2.COLOR_BGR2GRAY);
			grayResult = cv2.blur(grayResult,(3,3))
			grayResult = cv2.equalizeHist(grayResult)
			if(DEBUG):
				cv2.imshow('plate',grayResult)
				cv2.waitKey(0)
				cv2.destroyAllWindows()

			results.append(grayResult)

	return results