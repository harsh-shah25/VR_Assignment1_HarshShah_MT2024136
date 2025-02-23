
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
    
    base_f    = base.astype(np.float32)
    overlay_f = overlay.astype(np.float32)

    
    base_mask    = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)    > 0
    overlay_mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY) > 0

    
    
    
    dist_base    = cv2.distanceTransform(base_mask.astype(np.uint8), cv2.DIST_L2, 5)
    dist_overlay = cv2.distanceTransform(overlay_mask.astype(np.uint8), cv2.DIST_L2, 5)

    
    
    combined_dist = dist_base + dist_overlay + 1e-7
    alpha_overlay = dist_overlay / combined_dist
    alpha_base    = dist_base    / combined_dist

    
    alpha_overlay_3 = np.dstack([alpha_overlay]*3)
    alpha_base_3    = np.dstack([alpha_base]*3)

    
    blended = base_f * alpha_base_3 + overlay_f * alpha_overlay_3

    return blended.astype(np.uint8)


def place_image_on_canvas(canvas, image, offset):

    if canvas is None or image is None:
        print("Error: Canvas or image is None.")
        return None

    
    canvas_h, canvas_w = canvas.shape[:2]
    img_h, img_w = image.shape[:2]
    x_offset, y_offset = offset

    
    x_end = min(x_offset + img_w, canvas_w)
    y_end = min(y_offset + img_h, canvas_h)

    
    if x_offset >= canvas_w or y_offset >= canvas_h:
        print("Error: Offset is outside the canvas bounds.")
        return canvas

    
    canvas[y_offset:y_end, x_offset:x_end] = image[0:y_end - y_offset, 0:x_end - x_offset]

    return canvas






warped_left, H_left = stitch_images(img2, img1)
warped_right, H_right = stitch_images(img2,img3)
h, w = warped_left.shape[:2]

canvas = np.ones((h*2, w*3, 3), dtype=np.uint8)  
h_ref, w_ref = img2.shape[:2]
ref_corners = np.array([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float32).reshape(-1, 1, 2)

left_corners = warp_corners(img1, H_left)
right_corners = warp_corners(img3, H_right)


all_corners = np.concatenate((ref_corners, left_corners, right_corners), axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)


translation = np.array([
    [1, 0, -x_min],
    [0, 1, -y_min],
    [0, 0, 1]
])
canvas_width = x_max - x_min
canvas_height = y_max - y_min


warped_left = cv2.warpPerspective(img1, translation.dot(H_left), (canvas_width, canvas_height))
warped_right = cv2.warpPerspective(img3, translation.dot(H_right), (canvas_width, canvas_height))
panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
panorama[-y_min:h_ref - y_min, -x_min:w_ref - x_min] = img2

panorama = feather_blend_two_images(panorama, warped_left)
panorama = feather_blend_two_images(panorama, warped_right)


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


plt.subplot(2,3,5)
plt.title("PANAROMA1")
plt.imshow(cv2.cvtColor(panorama,cv2.COLOR_BGR2RGB))


plt.show()

