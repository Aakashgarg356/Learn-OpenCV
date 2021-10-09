import cv2 as cv
import numpy as np

# Color Spacing
# img = cv.imread("C:\\Users\\Aakash Garg\\Documents\\Learn-OpenCV\\Aakash.jpg")
# resized = cv.resize(img, (400, 500) , interpolation=cv.INTER_AREA)
# cv.imshow("Resized", resized)

# gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
# cv.imshow("Gray", gray)
# hsv = cv.cvtColor(resized, cv.COLOR_BGR2HSV)
# cv.imshow("HSV", hsv)
# lab = cv.cvtColor(resized, cv.COLOR_BGR2Lab)
# cv.imshow("LAB", lab)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Color Channels
# img = cv.imread("C:\\Users\\Aakash Garg\\Documents\\Learn-OpenCV\\Aakash.jpg")
# resized = cv.resize(img, (400, 500) , interpolation=cv.INTER_AREA)
# cv.imshow("Resized", resized)

# blank = np.zeros(resized.shape[:2], dtype='uint8')
# b, g, r = cv.split(resized)
# blue = cv.merge([b, blank, blank])
# green = cv.merge([blank, g, blank])
# red = cv.merge([blank, blank, r])
# cv.imshow("BLUE", blue)
# cv.imshow("Green", green)
# cv.imshow("Red", red)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Blurring and Smoothing
# img = cv.imread("C:\\Users\\Aakash Garg\\Documents\\Learn-OpenCV\\Aakash.jpg")
# resized = cv.resize(img, (400, 500) , interpolation=cv.INTER_AREA)
# cv.imshow("Resized", resized)
# average = cv.blur(resized, (3,3))
# cv.imshow("Average", average)

# Gaussian Blur
# blur = cv.blur(resized, (3,3), 0)
# cv.imshow("Gaussian", blur)

# Median Blur
# median = cv.medianBlur(resized, 3)
# cv.imshow("Median", median)

# Bilateral Blurring
# bilateral = cv.bilateralFilter(resized, 50, 50, 95)
# cv.imshow("Bilateral", bilateral)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Bitwise Operator
# blank = np.zeros((400, 400), dtype='uint8')
# cv.imshow("Blank", blank)
#
# rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
# cv.imshow("Rectangle", rectangle)

# circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)
# cv.imshow("Circle", circle)
#
# bit_and = cv.bitwise_and(rectangle, circle)
# cv.imshow("And", bit_and)
#
# bit_or = cv.bitwise_or(rectangle, circle)
# cv.imshow("OR", bit_or)
#
# bit_xor = cv.bitwise_xor(rectangle, circle)
# cv.imshow("XOR", bit_xor)
#
# bit_not = cv.bitwise_not(rectangle, circle)
# cv.imshow("Not", bit_not)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Masking
# video = cv.VideoCapture(0)
# while True:
#     count, frame = video.read()
#     blank = np.zeros(frame.shape[:2], dtype='uint8')
#     circle = cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 180, 255, -1)
#     cv.putText(blank, "Aakash Garg", (200, 450), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=(255, 255, 255))
#     mask = cv.bitwise_and(frame, frame, mask=circle)
#     cv.imshow("Masked", mask)
#     if cv.waitKey(1) & 0xFF==ord('q'):
#         break
# video.release()
# cv.destroyAllWindows()

# Computing Histograms
import matplotlib.pyplot as plt
# img = cv.imread("C:\\Users\\Aakash Garg\\Documents\\Learn-OpenCV\\Aakash.jpg")
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Masking
# blank = np.zeros(img.shape[:2], dtype='uint8')
# circle = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
# bit_and = cv.bitwise_and(gray, gray, mask=circle)
# cv.imshow("Bitwise And", bit_and)

# # GrayScale Histogram
# gray_hist = cv.calcHist([gray], [0], bit_and, [256], [0, 256])
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# no of pixels")
# plt.plot(gray_hist)
# plt.xlim([0, 256])
# plt.show()

# # Color Histogram
# plt.figure()
# plt.title("Colour Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# no of pixels")
# colors = ('b', 'g', 'r')
# for i, col in enumerate(colors):
#     hist = cv.calcHist([img], [i], None, [256], [0, 256])
#     plt.plot(hist, color=col)
#     plt.xlim([0, 256])

# plt.show()
# cv.waitKey(0)
# cv.destroyAllWindows()

# Face Detection and Eye Detection
# face_classifier = cv.CascadeClassifier("C:\\Users\\Aakash Garg\\Documents\\Learn-OpenCV\\face_classifier.xml")
# eye_classifier = cv.CascadeClassifier("C:\\Users\\Aakash Garg\\Documents\\Learn-OpenCV\\eye_classifier.xml")
# video = cv.VideoCapture(0)
# while True:
#     count, frame = video.read()
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 2)
#     for (x, y, w, h) in faces:
#         cv.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 100), 1)
#         roi_gray = gray[x:x+w, y:y+h]
#         roi_color = frame[x:x+w, y:y+h]
#         eyes = eye_classifier.detectMultiScale(roi_gray)
#         for (ex, ey, ew, eh) in eyes:
#             cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (11, 222, 243), 1)
#     cv.imshow("Original", frame)
#     if cv.waitKey(1) & 0xFF==ord('q'):
#         break
# video.release()
# cv.destroyAllWindows()

# THRESHOLDING
# img = cv.imread("C:\\Users\\Aakash Garg\\Documents\\Learn-OpenCV\\Aakash.jpg")
# cv.imshow("Original", img)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("Gray", gray)

# # Simple Thresholding
# threshold, thresh = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)
# cv.imshow("Threshold", thresh)

# # Adaptive Thresholding
# adaptive = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 0)
# cv.imshow("Adaptive", adaptive)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Edge Detection
# img = cv.imread("C:\\Users\\Aakash Garg\\Documents\\Learn-OpenCV\\Aakash.jpg")
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # Laplacian
# lap = cv.Laplacian(gray, cv.CV_64F)
# lap = np.uint8(np.absolute(lap))
# cv.imshow("Lap", lap)

# # Sobel
# sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
# sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
# merged = cv.bitwise_or(sobelx, sobely)
# cv.imshow("Sobelx", sobelx)
# cv.imshow("Sobely", sobely)
# cv.waitKey(0)
# cv.destroyAllWindows()