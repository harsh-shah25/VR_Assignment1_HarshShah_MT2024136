
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_match_features(img1, img2):

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # print("len of matches", len(matches))
    # print("type of matches[0]", (help(matches[0][0])))


    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches

def stitch_images(reference_img, warping_img):
    keypoints1, keypoints2, good_matches = detect_and_match_features(reference_img, warping_img)

    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        height, width, _ = warping_img.shape
        warped_img = cv2.warpPerspective(warping_img, H, (width*2, height))
        # warped_img = np.zeros((height,width*2,3),np.uint8)

        # warped_img[0:height, 0:width] = reference_img
        # warped_img = warping_img


        return warped_img
    else:
        print("Not enough matches found!")
        return None

img1 = cv2.imread("panorama/2pan1.jpeg")
img2 = cv2.imread("panorama/2pan2.jpeg")
img3 = cv2.imread("panorama/2pan3.jpeg")

panorama1 = stitch_images(img2, img1)
panorama2 = stitch_images(img2,img3)

if panorama1 is not None and panorama2 is not None:

    panorama_final = stitch_images(img3, panorama1)

    if panorama_final is not None:
        cv2.imwrite("panorama.jpg", panorama_final)

        plt.figure(figsize=(30,10))

        plt.subplot(2,3,1)
        plt.title("Image 1")
        plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))

        plt.subplot(2,3,2)
        plt.title("Image 2")
        plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))

        plt.subplot(2,3,3)
        plt.title("Image 3")
        plt.imshow(cv2.cvtColor(img3,cv2.COLOR_BGR2RGB))

        plt.subplot(2,3,4)
        plt.title("PANAROMA1")
        plt.imshow(cv2.cvtColor(panorama1,cv2.COLOR_BGR2RGB))

        plt.subplot(2,3,5)
        plt.title("PANAROMA2")
        plt.imshow(cv2.cvtColor(panorama2,cv2.COLOR_BGR2RGB))

        plt.subplot(2,3,6)
        plt.title("PANAROMA")
        plt.imshow(cv2.cvtColor(panorama_final,cv2.COLOR_BGR2RGB))

        plt.show()
    else:
        print("Image stitching failed for third image!")
else:
    print("Image stitching failed for first two images!")
