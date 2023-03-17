import argparse
import cv2
import imutils
from imutils.perspective import four_point_transform
from skimage.filters import threshold_local

# Creates the argument parser and parses the args
argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--image", required=True, help="Path to input image")
args = vars(argParser.parse_args())

# Loads the image
img = cv2.imread(args["image"])
# Calculate the img ratio
ratio = img.shape[0] / 500.0
# Clone of original
original = img.copy()
# Resize to new height
img = imutils.resize(img, height = 500)

# Converts the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blurs the grayscale image
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# Finds edges in the image
edges = cv2.Canny(gray, 75, 200)

print("STEP 1: Edge Detection")
cv2.imshow("Image", img)
cv2.imshow("Edged", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Finds contours in the edged img and displays largest ones
contours = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

for con in contours:
    perimeter = cv2.arcLength(con, True)
    approx = cv2.approxPolyDP(con, 0.02 * perimeter, True)

    # Looks for 4 points (like a square or rectangle)
    if len(approx) == 4:
        screenContour = approx
        break

print("STEP 2: Find Contours of Document")
cv2.drawContours(img, [screenContour], -1, (0, 255, 0), 2)
cv2.imshow("Outline", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Four Point Transform to get Top-Down View
topDown = four_point_transform(original, screenContour.reshape(4, 2) * ratio)

# Convert to Grayscale and Threshold for Black/White Effect
topDown = cv2.cvtColor(topDown, cv2.COLOR_BGR2GRAY)
threshold = threshold_local(topDown, 11, offset=10, method="gaussian")
topDown = (topDown > threshold).astype("uint8") * 255

print("STEP 3: Apply Perspective Transformation")
cv2.imshow("Original", imutils.resize(original, height=650))
cv2.imshow("Scanned", imutils.resize(topDown, height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()