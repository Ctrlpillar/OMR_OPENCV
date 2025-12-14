import cv2
import numpy as np
import utlis


########################################################################
pathImage = "1.jpg"
heightImg = 700
widthImg  =  700
questions = 5
choices = 5
ans = [1,2,0,1,4]
score = 0

########################################################################

img = cv2.imread(pathImage)
# PRE-PROCESSING THE IMAGE

img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgFinal = img.copy()
imgBiggestContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
imgCanny = cv2.Canny(imgBlur,10,50) # APPLY CANNY EDGE

# FIND CONTOURS
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
# FIND RECTANGLE
rectCon =utlis.rectContour(contours)
biggestContour = utlis.getCornerPoints(rectCon[0])

gradePoints = utlis.getCornerPoints(rectCon[1])
# print(biggestContour)

if biggestContour.size !=0 and gradePoints.size !=0:
    cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)

    biggestContour = utlis.reorder(biggestContour)
    gradePoints = utlis.reorder(gradePoints)

    pnt1 = np.float32(biggestContour) # PREPARE POINTS FOR WARP
    pnt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pnt1, pnt2) # GET TRANSFORMATION MATRIX
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # APPLY WARP PERSPECTIVE


    pntG1 = np.float32(gradePoints) # PREPARE POINTS FOR WARP
    pntG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]]) # PREPARE POINTS FOR WARP
    matrixG = cv2.getPerspectiveTransform   (pntG1, pntG2) # GET TRANSFORMATION MATRIX
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150)) # APPLY WARP PERSPECTIVE
    #cv2.imshow("Image Grade", imgGradeDisplay)



    #APPY THERE THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

    boxes = utlis.splitBoxes(imgThresh)
    # cv2.imshow("Test",boxes[0])
    # print(cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2]))

    #GETTING THE NON ZERO PIXEL VALUES OF EACH BOX
    myPixelValues = np.zeros((questions, choices))
    countC = 0
    countR = 0
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelValues[countR][countC] = totalPixels
        countC +=1
        if (countC==choices):countR +=1; countC=0
    #print(myPixelValues)


    #FINDING INDEX VALUES OF THE MAKRINGS
    myIndex = []
    for x in range (0, questions):
        arr = myPixelValues[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])
    #print(myIndex)
    

    #GRADING
    grading = []
    for x in range (0, questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
    #print(grading)
    score = (sum(grading)/questions)*100 #FInal Score

    print(score)

    # DISPLAYING ANSWERS
    imgResult = imgWarpColored.copy()
    imgResult = utlis.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
    imRawDrwaing = np.zeros_like(imgWarpColored)
    imRawDrwaing = utlis.showAnswers(imRawDrwaing, myIndex, grading, ans, questions, choices)
    invMatrix = cv2.getPerspectiveTransform(pnt2, pnt1) 
    imgInvWarp = cv2.warpPerspective(imRawDrwaing, invMatrix, (widthImg, heightImg))

    # DISPLAYING GRADE
    imgRawGrade = np.zeros_like(imgGradeDisplay)
    cv2.putText(imgRawGrade, str(int(score))+"%", (60,100), cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
    invMatrixG = cv2.getPerspectiveTransform(pntG2, pntG1) 
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))
    
    imgFinal = cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)
    imgFinal = cv2.addWeighted(imgFinal, 1 , imgInvGradeDisplay, 1, 0)







imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur,imgCanny],
              [imgContours, imgBiggestContours, imgWarpColored, imgThresh],
              [imgResult, imRawDrwaing, imgInvWarp, imgFinal])
lables=[["Original","Gray","Blur","Canny"],
         ["Contours","Biggest Contour","Warped","Threshold"],   
         ["Result","Raw Drawing","Inverse Warp","Final Result"]]
imgStacked = utlis.stackImages(imageArray,0.3,lables)

cv2.imshow("final Result", imgFinal)
cv2.imshow("Stacked Images", imgStacked)
cv2.waitKey(0)
