import os, random
import numpy as np
import cv2
import matplotlib.pyplot as plt

akaze = cv2.AKAZE_create()
bf    = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
DATA_PATH  = "../dataset/sequences/05"
CALIB_FILE = "../dataset/sequences/05/calib.txt"


def read_kitti_calib(calib_path):
    with open(calib_path) as f:
        lines = f.readlines()
    P0 = np.fromstring(lines[0].split(maxsplit=1)[1], sep=' ').reshape(3, 4)
    P1 = np.fromstring(lines[1].split(maxsplit=1)[1], sep=' ').reshape(3, 4)
    K  = P0[:, :3]
    return K, P0, P1


def read_images(idx, data_path=DATA_PATH):
    name = f"{idx:06d}.png"
    L = cv2.imread(os.path.join(data_path, "image_0", name), 0)
    R = cv2.imread(os.path.join(data_path, "image_1", name), 0)
    if L is None or R is None:
        raise FileNotFoundError("Bad DATA_PATH or index")
    return L, R


def triangulate_llsq(Pa, Pb, pt_a, pt_b):
    ua, va = pt_a;  ub, vb = pt_b
    A = np.vstack([ua*Pa[2]-Pa[0], va*Pa[2]-Pa[1],
                   ub*Pb[2]-Pb[0], vb*Pb[2]-Pb[1]])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1];          return X[:3] / X[3]


# intrinsics + stereo extrinsics
K, P0, P1 = read_kitti_calib(CALIB_FILE)

# -----------------------  point clouds  -----------------------
def build_point_cloud(idx, r_thresh=0.75, row_tol=2):
    L, R = read_images(idx)
    kpL, desL = akaze.detectAndCompute(L, None)
    kpR, desR = akaze.detectAndCompute(R, None)

    knn = bf.knnMatch(desL, desR, k=2)
    good = [m for m,n in knn if m.distance < r_thresh*n.distance]
    inliers = [m for m in good
               if abs(kpL[m.queryIdx].pt[1]-kpR[m.trainIdx].pt[1]) <= row_tol]

    cloud = np.array([triangulate_llsq(P0, P1,
                                       kpL[m.queryIdx].pt,
                                       kpR[m.trainIdx].pt)
                      for m in inliers])
    return L, R, kpL, kpR, inliers, cloud


L0, R0, kpL0, kpR0, inl0, cloud0 = build_point_cloud(0)
L1, R1, kpL1, kpR1, inl1, cloud1 = build_point_cloud(1)

print(f"Cloud-0: {len(cloud0)} pts   Cloud-1: {len(cloud1)} pts")

# map: key-point index → cloud row  (used later to fetch 3-D safely)
kpidx_to_cloudrow = {m.queryIdx: r for r, m in enumerate(inl0)}

#------------------------------------  left-left matches  -----------------------
desL0 = akaze.compute(L0, kpL0)[1]
desL1 = akaze.compute(L1, kpL1)[1]
knn_LL   = bf.knnMatch(desL0, desL1, k=2)
LL_matches = [m for m,n in knn_LL if m.distance < 0.75*n.distance]
print("left0↔left1 Lowe matches:", len(LL_matches))

# keep only matches whose queryIdx exists in the map (ensures 3-D point)
valid_LL = [j for j,m in enumerate(LL_matches) if m.queryIdx in kpidx_to_cloudrow]

# ----------------------- ------------------- minimal PnP + RANSAC -----------------------
def reproj_err(P, X, pix):
    x = P @ np.append(X,1);  x = x[:2]/x[2]
    return np.linalg.norm(x - pix)

def rt_to_mat(rvec, tvec):
    R,_ = cv2.Rodrigues(rvec);  return np.hstack((R, tvec))

def pnp_random4():
    ids = random.sample(valid_LL, 4)
    obj = np.float32([cloud0[kpidx_to_cloudrow[LL_matches[i].queryIdx]]
                      for i in ids])
    im  = np.float32([kpL1[LL_matches[i].trainIdx].pt for i in ids])
    ok,rvec,tvec = cv2.solvePnP(obj, im, K, None, flags=cv2.SOLVEPNP_EPNP)
    return ok, rt_to_mat(rvec, tvec)

def ransac_pnp(N=300, tol=2):
    best_sup, best_T = [], None
    for _ in range(N):
        ok,T = pnp_random4()
        if not ok: continue
        sup=[]
        for j in valid_LL:
            m = LL_matches[j]
            X = cloud0[kpidx_to_cloudrow[m.queryIdx]]
            if reproj_err(K@T, X, kpL1[m.trainIdx].pt) < tol:
                sup.append(j)
        if len(sup) > len(best_sup):
            best_sup, best_T = sup, T
    # refine
    obj_all = np.float32([cloud0[kpidx_to_cloudrow[LL_matches[i].queryIdx]]
                          for i in best_sup])
    img_all = np.float32([kpL1[LL_matches[i].trainIdx].pt for i in best_sup])
    _,rvec,tvec = cv2.solvePnP(obj_all, img_all, K, None,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    return best_sup, rt_to_mat(rvec, tvec)

inliers_LL, T01 = ransac_pnp()
print("PnP-RANSAC inliers:", len(inliers_LL))

# -------------------------- camera centres & four-camera plot
# camera centres
C_L0 = np.zeros(3)
Rt_R0 = np.linalg.inv(K) @ P1;  R_R0, t_R0 = Rt_R0[:,:3], Rt_R0[:,3]
C_R0  = -R_R0.T @ t_R0
R01, t01 = T01[:,:3], T01[:,3]
C_L1 = -R01.T @ t01
C_R1 = -R01.T @ (t01 + t_R0)

# top-view plot
pts = np.vstack([C_L0, C_R0, C_L1, C_R1])
names = ["left0","right0","left1","right1"]
plt.figure(figsize=(6,6))
plt.scatter(pts[:,0], pts[:,2], c=['r','b','g','c'])
for i,n in enumerate(names):
    plt.text(pts[i,0], pts[i,2], " "+n)
plt.axis('equal'); plt.grid(True)
plt.title("Four camera centres (top view)")
plt.xlabel("X  [m]"); plt.ylabel("Z  [m]")
plt.show()
