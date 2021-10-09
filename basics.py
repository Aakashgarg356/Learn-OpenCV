import cv2 as cv
import numpy as np

# Read an image
# img = cv.imread("C:\\Users\\Aakash Garg\\Documents\\Learn-OpenCV\\Aakash.jpg")
# cv.imshow("Aakash Garg", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Read a video
# video = cv.VideoCapture(0)
# while True:
#     count, frame = video.read()
#     cv.imshow("Video", frame)
#     if cv.waitKey(1) & 0xFF==ord('q'):
#         break
# video.release()
# cv.destroyAllWindows()

# Resize an image
# img = cv.imread("C:\\Users\\Aakash Garg\\Documents\\Learn-OpenCV\\Aakash.jpg")

# def resize_image(im, scale):
#     width = int(im.shape[1] * scale)
#     height = int(im.shape[0] * scale)
#     dimensions = (width, height)
#     return cv.resize(im, dimensions, interpolation=cv.INTER_AREA)
# resized = resize_image(img, 0.50)
# cv.imshow("Aakash Garg", resized)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Resize a video
# def resize_video(vi, scale):
#     width = int(vi.shape[1] * scale)
#     height = int(vi.shape[0] * scale)
#     dimensions = (width, height)
#     return cv.resize(vi, dimensions, interpolation=cv.INTER_AREA)

# video = cv.VideoCapture(0)
# while True:
#     count, frame = video.read()
#     resized = resize_video(frame, 0.50)
#     cv.imshow("Original Video", frame)
#     cv.imshow("Resized Video", resized)
#     if cv.waitKey(1) & 0xFF==ord('q'):
#         break
# video.release()
# cv.destroyAllWindows()

# OpenCV Functions
# def resize_image(im, scale):
#     width = int(im.shape[1] * scale)
#     height = int(im.shape[0] * scale)
#     dimensions = (width, height)
#     return cv.resize(im, dimensions, interpolation=cv.INTER_AREA)
# img = cv.imread("C:\\Users\\Aakash Garg\\Documents\\Learn-OpenCV\\Aakash.jpg")
# resized = resize_image(img, 0.50)
# cv.imshow("Aakash Garg", resized)

# gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
# cv.imshow("Gray Image", gray)

# blur = cv.GaussianBlur(gray, (7,7), cv.BORDER_DEFAULT)
# cv.imshow("Blur Video", blur)

# canny = cv.Canny(resized, 50, 100)
# cv.imshow("Canny", canny)

# dilation = cv.dilate(canny, (7,7), iterations=5)
# cv.imshow("Dilated Image", dilation)

# erosion = cv.erode(dilation, (7,7), iterations=1)
# cv.imshow("Eroted Image", erosion)

# cv.waitKey(0)
# cv.destroyAllWindows()

# Draw Images
# img = cv.imread("C:\\Users\\Aakash Garg\\Documents\\Learn-OpenCV\\Aakash.jpg")
# resized = cv.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv.INTER_AREA)
# cv.imshow("Resized Image", resized)
# blank = np.zeros(resized.shape[:2], dtype='uint8')

# rect = cv.rectangle(blank.copy(), (50, 50), (350, 350), 255, 1)
# cv.imshow("Rectangle", rect)

# circle = cv.circle(blank.copy(), (blank.shape[1]//2, blank.shape[0]//2), 100, 255, 1)
# cv.imshow("Circle", circle)

# line = cv.line(blank.copy(), (10,10), (100, 200), 255, 1)
# cv.imshow("Line", line)

# text = cv.putText(blank.copy(), "OpenCv is 101...", (90, 150), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color = (255, 222, 100), thickness=1)
# cv.imshow("Text", text)

# cv.waitKey(0)
# cv.destroyAllWindows()

# Image Transformation
# img = cv.imread("C:\\Users\\Aakash Garg\\Documents\\Learn-OpenCV\\Aakash.jpg")
# resized = cv.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv.INTER_AREA)
# cv.imshow("Resized Image", resized)

# def translate(im, x, y):
#     trans_matrix = np.float32([[1, 0, x], [0, 1, y]])
#     dimension = (resized.shape[1], resized.shape[0])
#     return cv.warpAffine(im, trans_matrix, dimension)

# translate_image = translate(resized, 100, -100)
# cv.imshow("Translated Image", translate_image)

# def rotated_image(ima, angle, rotPoint=None):
#     (height, width) = resized.shape[:2]
#     if rotPoint=="None":
#         rotPoint = (width//2, height//2)

#     rot_matrix =  cv.getRotationMatrix2D(rotPoint, angle, scale=1.0)
#     dimensions = (width, height)
#     return cv.warpAffine(ima, rot_matrix, dimensions)

# rotated = rotated_image(resized, 10)
# cv.imshow("Rotated Image", rotated)

# flip = cv.flip(resized, 1)
# cv.imshow("Flipped", flip)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Contours
# img = cv.imread("C:\\Users\\Aakash Garg\\Documents\\Learn-OpenCV\\Aakash.jpg")

# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# cv.imshow("Thresh", thresh)

# blank = np.zeros(img.shape, dtype='uint8')

# canny = cv.Canny(img, 50, 100)

# contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# print(len(contours))

# cv.drawContours(blank, contours, -1, (0, 255, 0), 1)
# cv.imshow("Contours", blank)

# cv.waitKey(0)
# cv.destroyAllWindows()