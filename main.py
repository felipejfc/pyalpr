import cv2

img = cv2.imread('resources/plates/placa8.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray,(3,3))
#img_sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3, 1, 1, cv2.BORDER_DEFAULT)
img_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

cv2.imshow('image',img_threshold[1])
cv2.waitKey(0)
cv2.destroyAllWindows()