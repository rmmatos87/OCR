# source: https://nanonets.com/blog/ocr-with-tesseract/#opensourceocrtools

import cv2
import pytesseract
import os
import numpy as np

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
# erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
     if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

def draw_boxes(img):
    h, w, c = img.shape
    boxes = pytesseract.image_to_boxes(img) 
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)

def get_image(path, name):
    return cv2.imread(path + "\\" + name)

if __name__ == "__main__":
    path = input("Path:")
    ls = [i for i in os.listdir(path) if "MicrosoftTeams-image" in i]
    custom_config = r'--oem 3 --psm 6'
    out = ""
    for i, name in enumerate(ls):
        print(i, " / ", len(ls))
    out += pytesseract.image_to_string(get_image(name), config=custom_config) + '\n'
    print(out)
