# Stereo outliers and triangulation

Made by Mohammad Awawda.

The script does the following:

• **Vertical-deviation statistics**
  – Computed the row difference delta v = v_L − v_R for every raw match.
  – Drew a histogram of Δv values with `matplotlib.pyplot.hist`.
  – Printed the percentage of matches whose |Δv| > 2 px, confirming that most correct correspondences respect the rectified-stereo constraint.

• **Geometric outlier rejection & visualisation**
  – Classified matches as **inliers** (|delta v| ≤ 2 px) or **outliers** (|delta v| > 2 px).
  – Rendered all matches on a side-by-side image: inliers in **orange**, outliers in **cyan**.
  – Reported the numbers kept/discarded.
  – Calculated the theoretical acceptance rate for random (uniform) erroneous matches
   and discussed why real scenes usually yield *more* surviving outliers.

• **Camera calibration import**
  – Parsed `calib.txt`, extracted projection matrices **P0**, **P1** and intrinsics **K**.
  – Normalised them so the left camera became \([I | 0]\).

• **Linear least-squares triangulation**
  – Implemented the 4×4 SVD method to triangulate every inlier pair.
  – Produced a 3-D scatter plot of the resulting point cloud (`mpl_toolkits.mplot3d`).

• **OpenCV cross-check**
  – Re-triangulated the same correspondences with `cv2.triangulatePoints`.
  – Computed and printed the **median Euclidean distance** between the two 3-D solutions to verify numerical consistency.
