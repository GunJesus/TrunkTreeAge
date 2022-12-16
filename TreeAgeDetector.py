import numpy as np
import matplotlib.pyplot as plt
import tifffile as tifffile
import cv2


# Func used to calc thresholds for Edge detection
def get_range(threshold, sigma=0.15):
    return (1 - sigma) * threshold, (1 + sigma) * threshold


# Load original Image
img = tifffile.imread('input.tif')
# Plot original Image
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()
# Apply log transform.
c = 255 / (np.log(1 + np.max(img)))
log_transformed = c * np.log(1 + img)
# Specify the data type.
img = np.array(log_transformed, dtype=np.uint8)
# Plot the current Image
img_plot = plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()
# Run Edge-Detection
img_blur = cv2.GaussianBlur(img, (3, 3), 0)
edge = cv2.Canny(img_blur, 20, 30)
# Calculate threshold
manual_thresh = np.median(img_blur)
manual_thresh = get_range(manual_thresh)
# Apply threshold
edges = cv2.Canny(img_blur, *manual_thresh)
# Plot progress
plt.imshow(edges, cmap='gray')
plt.show()
