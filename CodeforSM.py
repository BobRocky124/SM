import cv2
import numpy as np
from PIL import Image



# Load Images 
img1 = cv2.imread(r"C:\Users\Nafiz\Downloads\test1.jpg")
img2 = cv2.imread(r"C:\Users\Nafiz\Downloads\test2.jpg")

if img1 is None or img2 is None:
    raise ValueError("One of the images could not be loaded.")

# Resize second image to match first image
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Convert to grayscale makes things easier to compare for the computer
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#defect areas show bright color
diff = cv2.absdiff(gray1, gray2)

# highlight defects coloring places that are different 
_, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

#cleaning or something 
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

#  Apply Mask to Show Defects on Image 
defects_color = img1.copy()
defects_color[mask > 0] = [0, 0, 255]  

#  Save Results
cv2.imwrite("difference_gray.jpg", diff)
cv2.imwrite("defect_mask.jpg", mask)
cv2.imwrite("defects_overlay.jpg", defects_color)

# print to a file 
if np.sum(mask) == 0:
    print("No defects found")
else:
    print("Defects detected")

