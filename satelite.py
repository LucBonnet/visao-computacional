import math

import numpy as np
import cv2
import matplotlib.pyplot as plt

imagens = []

img = cv2.imread('./imagens/Satelite.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imagens.append(img)

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
maxGray = img_gray.max()
imagens.append(img_gray)

tamanhoKernel = 7
kernel = np.ones((tamanhoKernel,tamanhoKernel), np.uint8)

img_blur = cv2.blur(img_gray, ksize=(tamanhoKernel,tamanhoKernel))
img_blur2 = cv2.blur(img_blur, ksize=(tamanhoKernel,tamanhoKernel))
img_blur3 = cv2.blur(img_blur2, ksize=(tamanhoKernel,tamanhoKernel))
img_blur4 = cv2.blur(img_blur3, ksize=(tamanhoKernel,tamanhoKernel))
img_blur5 = cv2.blur(img_blur4, ksize=(tamanhoKernel,tamanhoKernel))
img_blur6 = cv2.blur(img_blur5, ksize=(tamanhoKernel,tamanhoKernel))
img_blur7 = cv2.blur(img_blur6, ksize=(tamanhoKernel,tamanhoKernel))

imagens.append(img_blur7)

img_grad = cv2.morphologyEx(img_blur7, cv2.MORPH_GRADIENT, kernel)
imagens.append(img_grad)

_, thresh = cv2.threshold(img_grad, maxGray *0.07, maxGray,cv2.THRESH_BINARY_INV)
imagens.append(thresh)

thresh_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
imagens.append(thresh_close)

img_d = cv2.dilate(thresh_close,kernel,iterations = 1)
imagens.append(img_d)

dilate_open = cv2.morphologyEx(img_d, cv2.MORPH_OPEN, kernel)
imagens.append(dilate_open)

contours, hierarchy = cv2.findContours(
                                   image = imagens[len(imagens)-1],
                                   mode = cv2.RETR_TREE,
                                   method = cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)
img_copy = img.copy()
final = cv2.drawContours(img_copy, contours, contourIdx = -1,
                         color = (255, 0, 0), thickness = 2)
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