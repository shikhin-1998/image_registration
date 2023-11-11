
"""
traditional feature based approaches:
1. keypoint detection
2. feature description 
3. feature matching
4. image warping

1. keypoint detection:
keypoint is a point of interest. each keypoint is represented by a descriptor. descriptor is feature vector contains keypoints charecteristics.
it should not change even if the scale, brightness .. changes

a. SIFT
b. SURF:
c. AKAZE
d. ORBFASTBRIEF

"""

"""
below we see implementation of AKAZE

"""

import numpy as np
import cv2

img1 = cv2.imread('lena.jpeg')
gray= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
akaze = cv2.AKAZE_create()
kp, descriptor = akaze.detectAndCompute(gray, None)
img1 = cv2.drawKeypoints(gray, kp, img1)
cv2.imshow("output", img1)
cv2.imwrite("akaze_output.jpg", img1)
# cv2.waitKey(0)

"""
once we identify keypoints from both the images, next we need to match keypoints from both the images
One method for this is BFMatcher.knnMatch()


"""

import numpy as np
import cv2

img1 = cv2.imread('lena.jpeg')
# img1 = cv2.resize(img1, (100, 100))
img1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('lena_r.jpeg')
# img2 = cv2.resize(img2, (100, 100))
img2= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
akaze = cv2.AKAZE_create()
# Find the keypoints and descriptors with SIFT
kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append([m])


# Draw matches
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('matches.jpg', img3)

cv2.imshow("output", img3)
# cv2.imwrite("akaze_output.jpg", img1)
# cv2.waitKey(0)

"""
if atleast 4 points are matching, we can tranform one image to another(relatively)
this is called image warping
for this we need to compute homography matrix
We use RANSAC algorithm to detect outliers and remove them before homography

we select good matches
then compute homography
finally warp image
"""

# Select good matched keypoints
ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])

# Compute homography
H, status = cv2.findHomography(sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC, 5.0)

# Warp image
warped_image = cv2.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))

cv2.imwrite('warped.jpg', warped_image)
cv2.imshow("warped output", warped_image)
cv2.waitKey(0)
