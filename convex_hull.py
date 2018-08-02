
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
#fgbg = cv2.createBackgroundSubtractorMOG2()

while 1:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #fgmask = fgbg.apply(blurred)

    _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
    thresh = cv2.Canny(thresh, 20, 100)

    kernel = np.ones((3,3),np.uint8)

    #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #erode = cv2.dilate(thresh, kernel, iterations = 1)

    im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)

    #boundingBoxes = [cv2.boundingRect(c) for c in contours]
    #(contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
    #key=lambda b:b[1][1], reverse=False))

    # dado um conjunto de pontos, dertemine o menor polígono/poliedro convexo contendo todos os pontos
    hull = [cv2.convexHull(c) for c in contours ]
    #hull = cv2.convexHull(contours[0])
    final = cv2.drawContours(frame, hull, -1, (0, 255, 255))

    cv2.imshow("frame", final)
    cv2.imshow("frame_2", thresh)

    if (cv2.waitKey(1) & 0xff) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""
import cv2
import numpy as np

img = cv2.imread('braco.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_OTSU) #cv2.THRESH_BINARY
im2, contours, hierarchy = cv2.findContours(thresh, 2, 1)
cnt = contours[0]

hull = cv2.convexHull(cnt, returnPoints = False)
defects = cv2.convexityDefects(cnt, hull)

for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img,start,end,[0,255,0],2)
    cv2.circle(img,far,5,[0,0,255],-1)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
