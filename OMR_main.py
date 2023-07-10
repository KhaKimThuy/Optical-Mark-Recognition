import cv2
import numpy as np
import utils

# path = r"D:\NCKH\sunny.png"
path = '1.png'
widthImg = 700
heightImg = 700
questions = 5
choices = 5
ans = [1,2,0,1,4]

img = cv2.imread(path)

# PREPROCESSING
img = cv2.resize(img, (widthImg, heightImg))

imgContours = img.copy()
imgBiggestContours = img.copy()
imgFinal = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

# FINDING ALL CONTOURS
contours, hier = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0,255,0), 10)

# FINDING RECTANGLES
rectCon = utils.rectContour(contours)
biggestContour = utils.getCornerPoints(rectCon[0])
gradePoints = utils.getCornerPoints(rectCon[1])

if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours, biggestContour, -1, (0,255,0), 20)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255,0,0), 20)

    biggestContour = utils.reorder(biggestContour)
    gradePoints = utils.reorder(gradePoints)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0,0], [325, 0], [0, 150], [325, 150]])
    matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

    # APPLY THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

    boxes = utils.splitBoxes(imgThresh)

    # GETTING NON ZERO PIXEL VALUE OF EACH BOX 
    myPixelVal = np.zeros((questions, choices))
    countR, countC = 0, 0
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if countC == choices:
            countR += 1
            countC = 0

    # FINDING INDEX VALUE OF MARKINGS    
    myIndex = []
    for x in range(questions):
        arr = myPixelVal[x]
        myIndexVal = np.argmax(arr)
        myIndex.append(myIndexVal)

    # GRADING
    grading = np.zeros_like(ans)
    for x in range(choices):
        if myIndex[x] == ans[x]:
            grading[x] = 1

    # FINAL GRADE
    score = sum(grading/questions)*100

    # DISPLAYING ANSWERS
    imgResult = imgWarpColored.copy()
    imgResult = utils.showAnswer(imgResult, myIndex, grading, ans, questions, choices)
    imgRawDrawing = np.zeros_like(imgWarpColored)
    imgRawDrawing = utils.showAnswer(imgRawDrawing, myIndex, grading, ans, questions, choices)
    invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
    imgInvWarp = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))

    imgRawGrade = np.zeros_like(imgGradeDisplay)
    cv2.putText(imgRawGrade, f'{int(score)}%', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), 3)
    invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
    imgInvWarpG = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
    imgFinal = cv2.addWeighted(imgFinal, 0.5, imgInvWarpG, 1, 0)

imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny],
              [imgContours, imgBiggestContours, imgWarpColored, imgThresh],
              [imgResult, imgRawDrawing, imgInvWarp, imgFinal])
labels = [["Original", "Gray", "Blur", "Canny"],
          ["Contours", "Biggest Con", "Warp", "Threshold"],
          ["Result", "Raw Drawing", "Inv Warp", "Final"]]
imgStacked = utils.stackImages(imageArray, 0.3)
cv2.imshow("Stacked Image", imgStacked)
cv2.waitKey(0)