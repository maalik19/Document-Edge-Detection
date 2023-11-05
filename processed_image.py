import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import os 

def mapp(h):
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

image = cv2.imread(r"C:\Users\maali\OneDrive\Bureau\test invoice\4.jpg")
image = cv2.resize(image, (1300, 800)) # redimensionner car V ne fonctionne pas bien avec de plus grandes images
orig = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()

blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
plt.imshow(blurred, cmap='gray')
plt.axis('off')
plt.show()

edged = cv2.Canny(blurred, 30, 50)  
plt.imshow(edged, cmap='gray')
plt.axis('off')
plt.show()

contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# la boucle extrait les contours de contour de la page
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)

    if len(approx) == 4:
        target = approx
        break
approx = mapp(target)  # trouver les points d'extrémité de la feuille

pts = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])  

op = cv2.getPerspectiveTransform(approx, pts) 
dst = cv2.warpPerspective(orig, op, (800, 800))

# Convertir l'image en niveaux de gris
gray_image = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

# Normaliser les valeurs des pixels entre 0 et 1
normalized_image =  gray_image / 255.0

# rendre l'écriture en gras
kernel = np.ones((3, 3), np.uint8)
eroded_image = cv2.erode(normalized_image, kernel, iterations=1)

# Filtre de débruitage
#reprocess_image = cv2.fastNlMeansDenoising(eroded_image, None, 50, 7, 21)

plt.imshow(normalized_image, cmap='gray')
plt.axis('off')
plt.show()

# Saving the processed image
#save_path = r"C:\Users\maali\OneDrive\Bureau\test invoice\processed_image33.jpg"
#cv2.imwrite(save_path, (eroded_image * 255).astype(np.uint8))



