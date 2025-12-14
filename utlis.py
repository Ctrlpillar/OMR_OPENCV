import cv2
import numpy as np


## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray, scale, labels=None):
    """Stack images into a single image.

    - Accepts a 2D list (rows x cols) or a 1D list of images.
    - Scales images uniformly and converts grayscale to BGR.
    - Replaces missing/invalid entries with a blank image of the target size.
    """
    if labels is None:
        labels = []

    rows = len(imgArray)
    cols = len(imgArray[0]) if isinstance(imgArray[0], list) else len(imgArray)
    rowsAvailable = isinstance(imgArray[0], list)

    # Find a reference width/height from the first valid image
    ref_w = ref_h = None
    if rowsAvailable:
        for r in imgArray:
            for im in r:
                if isinstance(im, np.ndarray):
                    ref_h, ref_w = im.shape[:2]
                    break
            if ref_w is not None:
                break
    else:
        for im in imgArray:
            if isinstance(im, np.ndarray):
                ref_h, ref_w = im.shape[:2]
                break

    if ref_w is None or ref_h is None:
        raise ValueError("No valid images found in imgArray to determine size")

    imageBlank = np.zeros((ref_h, ref_w, 3), np.uint8)

    def safe_resize(img):
        try:
            if not isinstance(img, np.ndarray):
                return imageBlank.copy()
            resized = cv2.resize(img, (0, 0), None, scale, scale)
            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            return resized
        except Exception:
            return imageBlank.copy()

    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                imgArray[x][y] = safe_resize(imgArray[x][y])

        hor = [np.hstack(imgArray[x]) for x in range(rows)]
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            imgArray[x] = safe_resize(imgArray[x])
        ver = np.hstack(imgArray)

    # Labels (optional)
    if labels:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(rows):
            for c in range(cols):
                text = labels[d][c]
                if not isinstance(text, str):
                    text = str(text)
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(text) * 13 + 27, 30 + eachImgHeight * d),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, text, (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

    return ver

def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2)) # REMOVE EXTRA BRACKET
    #print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    #print(add)
    #print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]

    return myPointsNew

def rectContour(contours):

    rectCon = []
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea,reverse=True)
    #print(len(rectCon))
    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx

def splitBoxes(img, rows=5, cols=5):
    """Split an image into (rows x cols) boxes and return a flat list of boxes."""
    r_parts = np.vsplit(img, rows)
    boxes = []
    for r in r_parts:
        c_parts = np.hsplit(r, cols)
        for box in c_parts:
            boxes.append(box)
    return boxes

def drawGrid(img, questions=5, choices=5):
    """Draw a grid with `questions` rows and `choices` columns on `img`."""
    secW = int(img.shape[1] / choices)
    secH = int(img.shape[0] / questions)

    # Horizontal lines (rows)
    for i in range(0, questions + 1):
        pt1 = (0, secH * i)
        pt2 = (img.shape[1], secH * i)
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)

    # Vertical lines (columns)
    for i in range(0, choices + 1):
        pt1 = (secW * i, 0)
        pt2 = (secW * i, img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)

    return img

def showAnswers(img, myIndex, grading, ans, questions, choices):
    secW = int(img.shape[1] / choices)
    secH = int(img.shape[0] / questions)

    for x in range(0, questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2
        if grading[x] == 1:
            myColor = (0, 255, 0)  # Green for correct
        else:
            myColor = (0, 0, 255)  # Red for incorrect
            correctAns = ans[x]
            cv2.circle(img, ((correctAns*secW)+secH//2,(x*secH)+secH//2), 20, (0,255,0), cv2.FILLED)
        cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)
    return img



