import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

PICUTRE_WIDTH=1126

# the provided code to set up the environment
DATA_PATH = "../dataset/sequences/05/"


def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    imgL = cv2.imread(DATA_PATH+'image_0/'+img_name, 0)
    imgR = cv2.imread(DATA_PATH+'image_1/'+img_name, 0)
    return imgL, imgR


# first initialize the AKAZE detector
akaze = cv2.AKAZE_create()

# load the first pair
img1, img2 = read_images(0)


def detect_and_match():
    global keypoints1, keypoints2, knn_matches, ratio_thresh, good_matches
    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = akaze.detectAndCompute(img1, None)
    keypoints2, descriptors2 = akaze.detectAndCompute(img2, None)
    # at least 500 keypoints are detected in each image
    if len(keypoints1) < 500 or len(keypoints2) < 500:
        print("Warning: Fewer than 500 keypoints detected in one or both images.")
    # Print the descriptors of the first two keypoints in each image
    print("Descriptors of the first two keypoints in the first image:")
    print(descriptors1[:2])
    print("\nDescriptors of the first two keypoints in the second image:")
    print(descriptors2[:2])
    # Initialize the Brute-Force Matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # Match descriptors between the two images using KNN that was briefly mentioned in the class
    knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # implementation of Lowe's test to filter matches(mentioned in wikipedia)
    ratio_thresh = 0.75
    good_matches = [m for m, n in knn_matches if m.distance < ratio_thresh * n.distance]


detect_and_match()


def draw_matches_before_ratio_test():
    if len(knn_matches) >= 20:
        random_matches = random.sample(knn_matches, 20)
        img_matches = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, random_matches, None,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(15, 7))
        plt.title('20 Random Matches Before Ratio Test')
        plt.imshow(img_matches)
        plt.show()


# Draw 20 random matches before applying the test
draw_matches_before_ratio_test()


def draw_matches_after_ratio_test():
    if len(good_matches) >= 20:
        random_good_matches = random.sample(good_matches, 20)
        img_good_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, random_good_matches, None,
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(15, 7))
        plt.title('20 Random Matches After Ratio Test')
        plt.imshow(img_good_matches)
        plt.show()
    # ratio value used and the number of matches discarded
    num_discarded = len(knn_matches) - len(good_matches)
    print(f"Ratio value used for Lowe's test: {ratio_thresh}")
    print(f"Number of matches discarded: {num_discarded}")
    # Create a list of "bad" matches (those that fail Lowe's ratio test)
    bad_matches = [m for m, n in knn_matches if m.distance >= ratio_thresh * n.distance]
    # Sample 20 random bad matches (if available)
    if len(bad_matches) >= 20:
        random_bad_matches = random.sample(bad_matches, 20)
    else:
        random_bad_matches = bad_matches
    # Draw the bad matches
    bad_match_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, random_bad_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 7))
    plt.title('Supposedly the Bad Matches')
    plt.imshow(bad_match_image)
    plt.show()


# Draw 20 random matches after applying the test
draw_matches_after_ratio_test()
