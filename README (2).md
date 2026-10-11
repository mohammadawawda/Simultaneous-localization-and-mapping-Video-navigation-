# Features and matching

Made by Mohammad Awawda.

The script does the following:

Image Loading: Reads a stereo image pair from a specified dataset.

Feature Extraction: Uses OpenCV's AKAZE to detect keypoints and compute descriptors for both images.

Matching: Performs KNN matching using BFMatcher and applies Lowe's ratio test to filter out ambiguous matches.

Bad Matches Identification: Extracts the "bad" matches (those failing the ratio test) and randomly samples 20 of them.

Visualization: Draws and displays these sampled bad matches using matplotlib.
