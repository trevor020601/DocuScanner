import argparse
import cv2
import imutils

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