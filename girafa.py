#pip install opencv-python

import math

import numpy as np
import cv2
import matplotlib.pyplot as plt

#Importa e converta para RGB
img = cv2.imread('./imagens/GIRAFA.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Convertendo para preto e branco (RGB -> Gray Scale -> BW)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
a = img_gray.max()

tamanhoKernel = 5
kernel = np.ones((tamanhoKernel,tamanhoKernel), np.uint8)
#Filtro de ruído (bluring)
img_blur = cv2.blur(img_gray, ksize=(tamanhoKernel,tamanhoKernel))
img_blur2 = cv2.blur(img_blur, ksize=(tamanhoKernel,tamanhoKernel))

_, thresh = cv2.threshold(img_blur2, a/2*1.7, a,cv2.THRESH_BINARY_INV)

thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

img_dilate = cv2.dilate(thresh_open,kernel,iterations = 2)
img_erode = cv2.erode(img_dilate,kernel,iterations = 1)

# contorno
contours, hierarchy = cv2.findContours(
                                   image = img_erode,
                                   mode = cv2.RETR_TREE,
                                   method = cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)
img_copy = img.copy()
final = cv2.drawContours(img_copy, contours, contourIdx = -1,
                         color = (255, 0, 0), thickness = 2)


#plot imagens
imagens = [img,img_blur,img_gray,thresh,thresh_open, img_dilate, img_erode, final]
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