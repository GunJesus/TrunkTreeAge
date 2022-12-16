import imutils
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

# Approximation of the middle Point
contours = cv2.findContours(trunk.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(contours)
c = max(cnts, key=cv2.contourArea)
# get all points
xs = c[:, 0, 0]
ys = c[:, 0, 1]
# get the maxima of each to draw lines and get the middle point --> this assumes the middle point
ymax = np.amax(ys)
index = np.where(ys == ymax)
ymax = np.array([xs[index][0], ymax])

ymin = np.amin(ys)
index = np.where(ys == ymin)
ymin = np.array([xs[index][0], ymin])

xmax = np.amax(xs)
index = np.where(xs == xmax)
xmax = np.array([xmax, ys[index][0]])

xmin = np.amin(xs)
index = np.where(xs == xmin)
xmin = np.array([xmin, ys[index][0]])


# calculate intersection
def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


# segment intersect and calculate with numpy
def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


# show intersects
xIntersec, yIntersec = seg_intersect(xmin, xmax, ymin, ymax)
points = np.array([xmin, xmax, ymin, ymax])
n = ["xmin", "xmax", "ymin", "ymax"]
plt.scatter(points[:, 0], points[:, 1])
for i, txt in enumerate(n):
    plt.annotate(txt, (points[i][0], points[i][1]))
plt.scatter(xIntersec, yIntersec)
plt.show()

# Map approxed middle to trunk
plt.gray()
plt.imshow(img)
plt.scatter(points[:, 0], points[:, 1])
plt.scatter(xIntersec, yIntersec)
plt.title("Calculated Middle Point of Tree")
plt.show()


# This algorithm copied from the internet to extract pixels along a line represented by two points
def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    # define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer


# Count the Rings
maskedImage = trunk
img_blur = cv2.GaussianBlur(maskedImage, (5, 5), 0)
edges = cv2.Canny(img_blur, 100, 200)


def counted_white_pixels(line_hor):
    count = 0
    li = createLineIterator(line_hor[0], line_hor[1], edges)

    for (r, g, b) in li:
        if b:
            count += 1
    return count


countedHor = counted_white_pixels((xmin, xmax))

countedver = counted_white_pixels((ymin, ymax))
print("Rings:")
# divided by 4 because we get over thw whole line --> meaning double the rings and every change is calculated so
# if it is from black to white and vise versa it counts as one, so we have double the change
# then we bild the mid around 4
print(int((countedver / 4 + countedHor / 4) / 2))

plt.imshow(edges)
