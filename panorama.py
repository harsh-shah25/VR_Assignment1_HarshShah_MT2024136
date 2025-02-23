
import cv2
import numpy as np
import matplotlib.pyplot as plt


def custom_resize(scale_percent,image):
    width = int(image.shape[1]*scale_percent/100)
    height = int(image.shape[0]*scale_percent/100)
    dim = (width,height)
    image = cv2.resize(image,dim)
    return image

def custom_resize_w(scale_percent,image):
    width = int(image.shape[1]*scale_percent/100)
    height = int(image.shape[0])
    dim = (width,height)
    image = cv2.resize(image,dim)
    return image

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

        H, _ = cv2.findHomography( dst_pts, src_pts,cv2.RANSAC, 5.0)

        height, width, _ = warping_img.shape
        warped_img = cv2.warpPerspective(warping_img, H, (width, height))
        # warped_img = np.zeros((height,width*2,3),np.uint8)

        # warped_img[0:height, 0:width] = reference_img
        # warped_img = warping_img


        return warped_img,H
    else:
        print("Not enough matches found!")
        return None

img1 = custom_resize(30,cv2.imread("panorama/1pan1.jpeg"))
img2 = custom_resize(30,cv2.imread("panorama/1pan2.jpeg"))
img3 = custom_resize(30,cv2.imread("panorama/1pan3.jpeg"))

def warp_corners(img, H):
    h, w = img.shape[:2]
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H)
    return warped


def feather_blend_two_images(base, overlay):
    """
    Feather-blend two images of the same shape in their overlapping region.
    base:    Numpy array (H x W x 3)
    overlay: Numpy array (H x W x 3)
             Must be in the same coordinate space as base.
    returns: Blended image of the same size
    """
    # Convert to float32 for safe blending
    base_f    = base.astype(np.float32)
    overlay_f = overlay.astype(np.float32)

    # Create masks where each image is non-zero
    base_mask    = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)    > 0
    overlay_mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY) > 0

    # Distance transform for each mask
    # The distance transform gives us, for each pixel, the distance to the nearest zero pixel
    # i.e. how deep into the valid region we are.
    dist_base    = cv2.distanceTransform(base_mask.astype(np.uint8), cv2.DIST_L2, 5)
    dist_overlay = cv2.distanceTransform(overlay_mask.astype(np.uint8), cv2.DIST_L2, 5)

    # Compute weights: alpha for the overlay, and 1 - alpha for the base in the overlap
    # We add a small epsilon (1e-7) to avoid division by zero in empty overlap.
    combined_dist = dist_base + dist_overlay + 1e-7
    alpha_overlay = dist_overlay / combined_dist
    alpha_base    = dist_base    / combined_dist

    # Expand alpha to 3 channels
    alpha_overlay_3 = np.dstack([alpha_overlay]*3)
    alpha_base_3    = np.dstack([alpha_base]*3)

    # Perform blending
    blended = base_f * alpha_base_3 + overlay_f * alpha_overlay_3

    return blended.astype(np.uint8)


def place_image_on_canvas(canvas, image, offset):
    """
    Places an image on a given canvas at the specified offset.

    :param canvas: The existing canvas (numpy array).
    :param image: The image to be placed (numpy array).
    :param offset: Tuple (x, y) specifying the top-left corner position on the canvas.
    :return: The updated canvas with the image placed at the offset.
    """
    if canvas is None or image is None:
        print("Error: Canvas or image is None.")
        return None

    # Get dimensions
    canvas_h, canvas_w = canvas.shape[:2]
    img_h, img_w = image.shape[:2]
    x_offset, y_offset = offset

    # Ensure image does not exceed canvas boundaries
    x_end = min(x_offset + img_w, canvas_w)
    y_end = min(y_offset + img_h, canvas_h)

    # Ensure offsets are within bounds
    if x_offset >= canvas_w or y_offset >= canvas_h:
        print("Error: Offset is outside the canvas bounds.")
        return canvas

    # Overlay the image onto the canvas
    canvas[y_offset:y_end, x_offset:x_end] = image[0:y_end - y_offset, 0:x_end - x_offset]

    return canvas

# Example usage
# offset = (100, 50)  # Place image at (x=100, y=50)

# updated_canvas = place_image_on_canvas(canvas, image, offset)

warped_left, H_left = stitch_images(img2, img1)
warped_right, H_right = stitch_images(img2,img3)
h, w = warped_left.shape[:2]
# print(h, w)
canvas = np.ones((h*2, w*3, 3), dtype=np.uint8)  # White canvas of 800x600
h_ref, w_ref = img2.shape[:2]
ref_corners = np.array([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float32).reshape(-1, 1, 2)

left_corners = warp_corners(img1, H_left)
right_corners = warp_corners(img3, H_right)

# Get overall bounds
all_corners = np.concatenate((ref_corners, left_corners, right_corners), axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# Compute translation matrix to ensure all coordinates are positive
translation = np.array([
    [1, 0, -x_min],
    [0, 1, -y_min],
    [0, 0, 1]
])
canvas_width = x_max - x_min
canvas_height = y_max - y_min

# Warp left and right images into the reference frame with translation
warped_left = cv2.warpPerspective(img1, translation.dot(H_left), (canvas_width, canvas_height))
warped_right = cv2.warpPerspective(img3, translation.dot(H_right), (canvas_width, canvas_height))

# Place the reference image on the canvas
panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
# Reference image position after translation (its top-left corner moves by (-x_min, -y_min))
panorama[-y_min:h_ref - y_min, -x_min:w_ref - x_min] = img2

# Overlay the warped left and right images (simply replace non-black pixels)
# mask_left = (warped_left > 0)
# panorama[mask_left] = warped_left[mask_left]

# mask_right = (warped_right > 0)
# panorama[mask_right] = warped_right[mask_right]
panorama = feather_blend_two_images(panorama, warped_left)
panorama = feather_blend_two_images(panorama, warped_right)


# place_image_on_canvas(canvas, warped_left, (0, 0))
# place_image_on_canvas(canvas, img2, (0, 0))
# plt.figure(figsize=(30,10))
# plt.show()
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

# plt.figure(figsize=(30, 30))
plt.subplot(2,3,5)
plt.title("PANAROMA1")
plt.imshow(cv2.cvtColor(panorama,cv2.COLOR_BGR2RGB))

# plt.subplot(2,3,5)
# plt.title("PANAROMA2")
# plt.imshow(cv2.cvtColor(panorama2,cv2.COLOR_BGR2RGB))

# plt.subplot(2,3,6)
# plt.title("PANAROMA")
# plt.imshow(cv2.cvtColor(panorama_final,cv2.COLOR_BGR2RGB))

plt.show()

"""
panorama1 = stitch_images(img2, img1)
panorama2 = panorama

if panorama1 is not None and panorama2 is not None:

    panorama_final = stitch_images(img3, panorama1)

    if panorama_final is not None:
        cv2.imwrite("panorama.jpg", panorama_final)

        
    else:
        print("Image stitching failed for third image!")
else:
    print("Image stitching failed for first two images!")
"""
# cv2.imshow("pano",panorama)
# cv2.waitKey()
# cv2.destroyAllWindows()