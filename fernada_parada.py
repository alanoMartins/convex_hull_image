import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    fgmask = fgbg.apply(blurred)
    ret, thresh = cv2.threshold(gray, 100, 180, cv2.THRESH_OTSU)


    kernel = np.ones((4,4),np.uint8)
    # dilated = cv2.dilate(thresh, kernel, iterations = 4)
    # erode = cv2.erode(fgmask, kernel, iterations = 1)
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(frame, contours, -1, (255,0,255), 3)
    # cnt = contours[4]
    # cv2.drawContours(img, [cnt], 0, (0,255,0), 3)



    if len(contours) > 0 :
# for c in contours:
# compute the center of the contour
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
		key=lambda b:b[1][1], reverse=False))

        c = contours[0]
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # draw the contour and center of the shape on the image
        cv2.drawContours(opening, [c], -1, (0, 255, 0), 2)
        cv2.circle(opening, (cX, cY), 7, (255, 0, 255), -1)
        cv2.putText(opening, "center", (cX - 20, cY - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    vertices = cv2.approxPolyDP(c, 2, True)
    vertices = cv2.convexHull(c, clockwise=True)
    # Display the resulting frame
    cv2.imshow('frame', opening)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
