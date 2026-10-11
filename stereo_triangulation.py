import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "../dataset/sequences/05/"
import os
# -------------------------------------------------------------------------
# Utility: load the left/right rectified images of frame `idx`
def read_images(idx, data_path=DATA_PATH):
    img_name = f"{idx:06d}.png"
    left  = os.path.join(data_path, "image_0", img_name)
    right = os.path.join(data_path, "image_1", img_name)
    imgL  = cv2.imread(left,  cv2.IMREAD_GRAYSCALE)
    imgR  = cv2.imread(right, cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        raise FileNotFoundError("Check DATA_PATH – could not read "
                                f"{left} or {right}")
    return imgL, imgR

# -------------------------------------------------------------------------
#   Load the first stereo pair (index 0)
imgL, imgR = read_images(0)

# -------------------------------------------------------------------------
#   Detect key-points & descriptors with AKAZE
akaze = cv2.AKAZE_create()
kpL, desL = akaze.detectAndCompute(imgL, None)
kpR, desR = akaze.detectAndCompute(imgR, None)
print(f"Detected {len(kpL)} key-points in the LEFT image, "
      f"{len(kpR)} in the RIGHT image.")

# -------------------------------------------------------------------------
#   Match all descriptors without using the significance test
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
knn_matches = bf.knnMatch(desL, desR, k=2)         # list[ (best, 2nd) ]
raw_matches  = [m[0] for m in knn_matches if len(m) > 0]
print(f"Total raw matches (no significance test): {len(raw_matches)}")

# -------------------------------------------------------------------------
#   row difference between the matches
#
dy = np.asarray([kpL[m.queryIdx].pt[1] - kpR[m.trainIdx].pt[1]
                 for m in raw_matches], dtype=np.float32)

# histogram
plt.figure(figsize=(8,4))
plt.hist(dy, bins=60, edgecolor="black")
plt.title(r"Histogram of vertical deviation  $\Delta v = v_L - v_R$")
plt.xlabel("deviation from rectified pattern  [pixels]")
plt.ylabel("number of matches")
plt.grid(alpha=0.4, linestyle="--")
plt.tight_layout()
plt.show()

# percentage deviating by more than +-2 px
TOL_PX = 2
pct_bad = (np.abs(dy) > TOL_PX).mean() * 100.0
print(f"{pct_bad:.2f}% of matches deviate by more than +-{TOL_PX} pixels.")


def reject_and_plot(
        imgL, imgR, kpL, kpR, raw_matches,
        row_tolerance_px=2, dot_radius=3):
    """
    Accepts / rejects matches via delta v <= row_tolerance_px, visualises the result
    and prints the statistics.
    """
    # ------------------------------------------------------------------
    # 1. split into inliers / outliers
    dy = np.asarray([kpL[m.queryIdx].pt[1] - kpR[m.trainIdx].pt[1]
                     for m in raw_matches])
    inlier_mask = np.abs(dy) <= row_tolerance_px
    inliers  = [m for m, keep in zip(raw_matches, inlier_mask) if keep]
    outliers = [m for m, keep in zip(raw_matches, inlier_mask) if not keep]

    # ------------------------------------------------------------------

    h, w = imgL.shape
    canvas = np.hstack((cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)))

    ORANGE = (0,165,255)   # BGR  (OpenCV default is BGR)
    CYAN   = (255,255,0)

    # draw inliers
    for m in inliers:
        uL, vL = kpL[m.queryIdx].pt
        uR, vR = kpR[m.trainIdx].pt
        cv2.circle(canvas, (int(uL),           int(vL)), dot_radius, ORANGE, -1)
        cv2.circle(canvas, (int(uR)+w,         int(vR)), dot_radius, ORANGE, -1)

    # draw outliers
    for m in outliers:
        uL, vL = kpL[m.queryIdx].pt
        uR, vR = kpR[m.trainIdx].pt
        cv2.circle(canvas, (int(uL),           int(vL)), dot_radius, CYAN, -1)
        cv2.circle(canvas, (int(uR)+w,         int(vR)), dot_radius, CYAN, -1)

    plt.figure(figsize=(12,5))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title("Matches:   inliers = orange   |   outliers = cyan")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    #  statistics
    n_total = len(raw_matches)
    n_out   = len(outliers)
    n_in    = len(inliers)
    print(f"Inliers kept : {n_in}")
    print(f"Outliers disc.: {n_out} ( {n_out/n_total*100:.1f}% of all matches )")

    # ------------------------------------------------------------------
    #  acceptance ratio under uniform error model
    H_px = imgL.shape[0]
    p_accept_uniform = (2*row_tolerance_px + 1) / H_px
    print("\nUniform-distribution thought experiment:")
    print(f"  Image height H            = {H_px} px")
    print(f"   tolerance              = ±{row_tolerance_px} px")
    print(f"  Theoretical accept ratio  = {p_accept_uniform*100:.2f}%")

    expected_bad_accepted = n_out * p_accept_uniform    # if all outliers were erroneous
    print(f"  ⇒ If all {n_out} discarded matches were erroneous,\n"
          f"     approx. {expected_bad_accepted:.1f} of them would have passed "
          f"the test by chance.\n")

    # ------------------------------------------------------------------
    print("Comment:")
    print("  The 'uniform' assumption is rarely true: false matches tend to "
          "cluster on\n  textured rows (road markings, windows, roof tiles) "
          "so their vertical disparity\n  is often smaller than random.  "
          "Hence the actual number of erroneous\n  matches that survive "
          "the diff <= 2 test is usually higher than the uniform-model\n  "
          "prediction shown above.\n")
    return inliers, outliers

inliers,outliers=reject_and_plot(imgL, imgR, kpL, kpR, raw_matches)


# -------------------------------------------------------------------------


# triangulation


# -------------------------------------------------------------------------

import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#  ------------  read the stereo calibration  ---------------------------------
def read_kitti_calib(calib_path):
    """
    Returns K (intrinsics), P0, P1   as 3×3, 3×4, 3×4 numpy arrays.
    """
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    # first two lines are "P0:" and "P1:"
    P0 = np.fromstring(lines[0].split(maxsplit=1)[1], sep=' ').reshape(3, 4)
    P1 = np.fromstring(lines[1].split(maxsplit=1)[1], sep=' ').reshape(3, 4)
    K  = P0[:, :3]                   # KITTI stores left/right with identical K
    return K, P0, P1

CALIB_FILE = "../dataset/sequences/05/calib.txt"
K, P0, P1 = read_kitti_calib(CALIB_FILE)

# normalise so that left camera becomes  [ I | 0 ]
P0n = np.linalg.inv(K) @ P0
P1n = np.linalg.inv(K) @ P1         # right camera in left-cam coords

#   ------------  helper: LLSQ triangulation of ONE match  ---------------------
def triangulate_llsq(Pa, Pb, pt_a, pt_b):
    """
    Linear least-squares triangulation
    pt_a / pt_b are (u,v) image coords in *pixels*.
    Returns 3-D point in the left-camera frame.
    """
    u1, v1 = pt_a
    u2, v2 = pt_b
    A = np.vstack([
        u1*Pa[2] - Pa[0],
        v1*Pa[2] - Pa[1],
        u2*Pb[2] - Pb[0],
        v2*Pb[2] - Pb[1]
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

#  ------------  triangulate every inlier match  ----------------------
pts_llsq = np.array([
    triangulate_llsq(P0n, P1n,
                     kpL[m.queryIdx].pt,
                     kpR[m.trainIdx].pt)
    for m in inliers   ])

#   ------------  3-D scatter plot  -------------------------------------------
fig = plt.figure(figsize=(7,6))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(pts_llsq[:,0], pts_llsq[:,1], pts_llsq[:,2], s=4)
ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
ax.set_title("3-D points via linear least-squares triangulation")
plt.tight_layout(); plt.show()

#   ------------  OpenCV  triangulation  ------------------------------
# prepare 2×N float32 matrices of pixel coords
pts_L = np.float32([kpL[m.queryIdx].pt for m in inliers]).T   # shape 2×N
pts_R = np.float32([kpR[m.trainIdx].pt  for m in inliers]).T

# OpenCV works in normalised coords too
pts_cv = cv2.triangulatePoints(P0n, P1n, pts_L, pts_R)        # 4×N
pts_cv = (pts_cv[:3] / pts_cv[3]).T                           # N×3

#   ------------  compare using median Euclidean distance  -------------------------
dists = np.linalg.norm(pts_llsq - pts_cv, axis=1)
print(f"Median between LLSQ − OpenCV and  distance = {np.median(dists):.4f} metres")
