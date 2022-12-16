import numpy as np
import matplotlib.pyplot as plt
import tifffile as tifffile
import cv2


# Func used to calc thresholds for Edge detection
def get_range(threshold, sigma=0.1):
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
thresh = np.median(img_blur)
thresh = get_range(thresh)
# Apply threshold
edges = cv2.Canny(img_blur, *thresh)
# Plot progress
plt.imshow(edges, cmap='gray')
plt.show()
# Apply dilation to make continues edges
image = img
gray = edges
# apply dilation on src image
kernel = np.ones((3, 3), np.uint8)
dilated_img = cv2.dilate(gray, kernel, iterations=39)
# blur image to remove white dots
img = cv2.medianBlur(dilated_img, 255)
# Plot new image
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()
# Invert colors
img = cv2.bitwise_not(img)
# Plot new image
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()

img = img.astype('uint8')
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
sizes = stats[:, -1]

max_label = 1
max_size = sizes[1]
for i in range(2, nb_components):
    if sizes[i] > max_size:
        max_label = i
        max_size = sizes[i]

img2 = np.zeros(output.shape)
img2[output == max_label] = 255
mask = img2.astype('uint8')

plt.imshow(img2, cmap=plt.get_cmap('gray'))
plt.show()
# Cut out Trunk
img = tifffile.imread('input.tif')
trunk = cv2.bitwise_and(img, img, mask=mask)
# Plot Trunk
plt.imshow(trunk, cmap=plt.get_cmap('gray'))
plt.show()
