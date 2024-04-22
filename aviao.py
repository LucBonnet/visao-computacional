import math

import numpy as np
import cv2
import matplotlib.pyplot as plt

imagens = []

img = cv2.imread('./imagens/Aviao.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imagens.append(img)

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
a = img_gray.max()
imagens.append(img_gray)

tamanhoKernel = 5
kernel = np.ones((tamanhoKernel,tamanhoKernel), np.uint8)
kernel2 = np.array([[1 if x == 2 else 0, 1, 1 if x == 2 else 0] for x in range(0,3)], np.uint8)

img_blur = cv2.blur(img_gray, ksize=(tamanhoKernel,tamanhoKernel))
img_blur2 = cv2.blur(img_blur, ksize=(tamanhoKernel,tamanhoKernel))
imagens.append(img_blur2)

_, thresh = cv2.threshold(img_blur2, a*0.21, a,cv2.THRESH_BINARY_INV)
imagens.append(thresh)

img_dilate = cv2.dilate(thresh,kernel,iterations = 4)
imagens.append(img_dilate)

img_dilate2 = cv2.dilate(img_dilate,kernel2,iterations = 5)
imagens.append(img_dilate2)

img_erode = cv2.erode(img_dilate2,kernel2,iterations = 4)
imagens.append(img_erode)

img_dilate3 = cv2.dilate(img_erode,kernel,iterations = 8)
imagens.append(img_dilate3)

img_erode2 = cv2.erode(img_dilate3,kernel,iterations = 8)
imagens.append(img_erode2)

# contorno
contours, hierarchy = cv2.findContours(
                                   image = imagens[len(imagens)-1],
                                   mode = cv2.RETR_TREE,
                                   method = cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)
img_copy = img.copy()
final = cv2.drawContours(img_copy, contours, contourIdx = -1, color = (255, 0, 0), thickness = 2)

imagens.append(final)

formatoX = math.ceil(len(imagens)**.5)
if (formatoX**2-len(imagens))>formatoX:
    formatoY = formatoX-1
else:
    formatoY = formatoX
for i in range(len(imagens)):
    plt.subplot(formatoY, formatoX, i + 1)
    plt.imshow(imagens[i],'gray')
    plt.xticks([]),plt.yticks([])
plt.show()

plt.imshow(imagens[len(imagens)-1],'gray')
plt.xticks([]),plt.yticks([])
plt.show()