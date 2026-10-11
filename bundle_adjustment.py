
import random, math, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam.utils.plot import plot_trajectory

from tracking_database import TrackingDB
from pnp_sequence import compute_global_extrinsics

# Define file paths
ROOT = Path(__file__).resolve().parent.parent
SEQ_PATH = ROOT / "dataset" / "sequences" / "05"
POSES_F = ROOT / "dataset" / "poses" / "05.txt"
CALIB_F = SEQ_PATH / "calib.txt"
PNP_DIR = ROOT / "out_pnp_and_ransac"
TRACKS_DIR = ROOT / "out_feature_tracking"
OUT_DIR = ROOT / "out_bundle_adjustment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Read calibration data for the KITTI sequence
def read_kitti_calib(calib_file: Path):
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    P0_str = lines[0].split(':')[1].strip()
    P1_str = lines[1].split(':')[1].strip()
    P0 = np.fromstring(P0_str, sep=' ').reshape(3, 4)
    P1 = np.fromstring(P1_str, sep=' ').reshape(3, 4)
    K = P0[:, :3]
    baseline = abs(P1[0, 3] / P1[0, 0])
    return K, P0, P1, baseline


K, P0, P1, base = read_kitti_calib(CALIB_F)
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
K_stereo = gtsam.Cal3_S2Stereo(fx, fy, 0.0, cx, cy, -base)

# Load the tracking database from feature tracking
DB_FILE = TRACKS_DIR / "tracks"
db = TrackingDB()
db.load(str(DB_FILE))
frames = sorted(db.all_frames())

# Compute or load global camera poses from PnP
PNP_FILE = PNP_DIR / "pnp_left_poses.npy"
if not PNP_FILE.exists():
    print("Computing PnP chain...")
    P_left = compute_global_extrinsics(db, K, P0, P1, frames)
    np.save(PNP_FILE, P_left)
else:
    P_left = np.load(PNP_FILE)

# Cache for computed GTSAM poses
_pose_cache = {}


def get_gtsam_pose(frame_id: int) -> gtsam.Pose3:
    if frame_id not in _pose_cache:
        R_cw, t_cw = P_left[frame_id, :3, :3], P_left[frame_id, :3, 3]
        R_wc, t_wc = R_cw.T, -R_cw.T @ t_cw
        _pose_cache[frame_id] = gtsam.Pose3(gtsam.Rot3(R_wc), gtsam.Point3(t_wc))
    return _pose_cache[frame_id]


def set_axes_equal_3d(ax):
    """Helper function to make 3D plot axes have equal scale."""
    x_lim, y_lim, z_lim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    x_range, x_mid = abs(x_lim[1] - x_lim[0]), np.mean(x_lim)
    y_range, y_mid = abs(y_lim[1] - y_lim[0]), np.mean(y_lim)
    z_range, z_mid = abs(z_lim[1] - z_lim[0]), np.mean(z_lim)
    radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - radius, x_mid + radius])
    ax.set_ylim3d([y_mid - radius, y_mid + radius])
    ax.set_zlim3d([z_mid - radius, z_mid + radius])


def reprojection_and_factor_error():
    print("\n--- Reprojection & Factor Error ---")
    for _ in range(300):
        tid = random.choice([t for t in db.all_tracks() if len(db.frames(t)) >= 10])
        track_frames = db.frames(tid)
        mid_idx = len(track_frames) // 2
        f_mid, mid_link = track_frames[mid_idx], db.link(track_frames[mid_idx], tid)
        try:
            cam_mid = gtsam.StereoCamera(get_gtsam_pose(f_mid), K_stereo)
            Pw = cam_mid.backproject(gtsam.StereoPoint2(mid_link.x_left, mid_link.x_right, mid_link.y))
            if np.any(np.isnan(Pw)): continue

            repro_errs, factor_errs, valid_frames = [], [], []
            for f in track_frames:
                try:
                    proj = gtsam.StereoCamera(get_gtsam_pose(f), K_stereo).project(Pw)
                    link_f = db.link(f, tid)
                    err = np.array([proj.uL() - link_f.x_left, proj.uR() - link_f.x_right, proj.v() - link_f.y])
                    repro_errs.append(np.linalg.norm(err))
                    factor_errs.append(np.dot(err, err))
                    valid_frames.append(f)
                except RuntimeError:
                    continue
            if len(valid_frames) >= 5: break
        except RuntimeError:
            continue
    else:
        print("-> Could not find a suitable track for analysis.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(valid_frames, repro_errs, 'o-');
    ax1.set_title(f"Reprojection L2 Error for Track {tid}");
    ax1.set_ylabel("Pixels");
    ax1.grid(True)
    ax2.plot(valid_frames, factor_errs, 's-r');
    ax2.set_title(f"Factor Chi-squared Error for Track {tid}");
    ax2.set_ylabel("χ²");
    ax2.set_xlabel("Frame ID");
    ax2.grid(True)
    fig.tight_layout();
    plt.savefig(OUT_DIR / "reprojection_errors.png");
    plt.close(fig)
    print(f"-> Saved error plots for track {tid}.")


def analytic_transformation():
    print("\n--- Analytic Transformation ---")
    T_w_A, T_w_B = get_gtsam_pose(0), get_gtsam_pose(20)
    T_A_B = T_w_A.between(T_w_B)
    print(f"Pose of B relative to A (T_A->B):\n{T_A_B}")

    T_B_A = T_A_B.inverse()
    R, t = T_B_A.rotation().matrix(), T_B_A.translation()
    extrinsic_B_in_A = np.hstack([R, t.reshape(3, 1)])
    print(f"\nExtrinsic matrix of B in frame A:\n{extrinsic_B_in_A}")


def select_keyframes():
    kfs, last_pose = [frames[0]], get_gtsam_pose(frames[0])
    DIST_THRESH, ROT_THRESH = 7.0, 0.05
    for f in frames[1:]:
        curr_pose = get_gtsam_pose(f)
        relative_pose = last_pose.between(curr_pose)
        dist = np.linalg.norm(relative_pose.translation())
        try:
            angle = relative_pose.rotation().axisAngle()[1]
        except:
            angle = 0
        if dist > DIST_THRESH or angle > ROT_THRESH:
            kfs.append(f)
            last_pose = curr_pose
    print(f"-> Selected {len(kfs)} keyframes.")
    return kfs


def setup_ba_graph(f0, f1, use_relative_coords=False):
    graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05, 0.2, 0.2, 0.2]))
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    T_w_k0 = get_gtsam_pose(f0)

    for f in range(f0, f1 + 1):
        if f not in frames: continue
        init_pose = T_w_k0.inverse().compose(get_gtsam_pose(f)) if use_relative_coords else get_gtsam_pose(f)
        values.insert(gtsam.symbol('c', f), init_pose)

    anchor_pose = gtsam.Pose3() if use_relative_coords else T_w_k0
    graph.add(gtsam.PriorFactorPose3(gtsam.symbol('c', f0), anchor_pose, prior_noise))

    for tid in db.all_tracks():
        obs_in_window = [(f, db.link(f, tid)) for f in db.frames(tid) if f0 <= f <= f1]
        if len(obs_in_window) < 2: continue

        qk = gtsam.symbol('q', tid)
        if values.exists(qk): continue

        f_last_obs, link_last = obs_in_window[-1]
        try:
            Pw = gtsam.StereoCamera(get_gtsam_pose(f_last_obs), K_stereo).backproject(
                gtsam.StereoPoint2(link_last.x_left, link_last.x_right, link_last.y))
            if np.any(np.isnan(Pw)): continue
        except RuntimeError:
            continue

        init_point = T_w_k0.inverse().transformFrom(Pw) if use_relative_coords else Pw
        values.insert(qk, init_point)

        for f_obs, link in obs_in_window:
            if values.exists(gtsam.symbol('c', f_obs)):
                meas = gtsam.StereoPoint2(link.x_left, link.x_right, link.y)
                graph.add(gtsam.GenericStereoFactor3D(meas, measurement_noise, gtsam.symbol('c', f_obs), qk, K_stereo))
    return graph, values


def first_ba_window():
    print("\n--- First BA Window Analysis ---")
    kfs = select_keyframes()
    f0, f1 = kfs[0], kfs[1]
    graph, vals = setup_ba_graph(f0, f1)
    if graph.size() < 10:
        print("-> First BA window has too few factors.");
        return

    init_err = graph.error(vals)
    print(f"Window [{f0},{f1}]: {graph.size()} factors, Initial Error={init_err:.1f}")

    params = gtsam.LevenbergMarquardtParams();
    params.setVerbosityLM("SILENT")
    result = gtsam.LevenbergMarquardtOptimizer(graph, vals, params).optimize()
    print(f"Final Error: {graph.error(result):.1f}")

    fig = plt.figure(figsize=(8, 7));
    ax = fig.add_subplot(111, projection='3d')
    plot_trajectory(fig, result, scale=1);
    ax.set_title("5.3: Optimized BA Trajectory")
    set_axes_equal_3d(ax);
    plt.savefig(OUT_DIR / "ba_window.png");
    plt.close(fig)
    print("-> Saved first BA window trajectory plot.")


def sliding_window_odometry():
    print("\n--- Sliding-Window Visual Odometry ---")
    kfs = select_keyframes()
    est_poses, gt_poses = [get_gtsam_pose(kfs[0])], [get_gtsam_pose(f) for f in kfs]

    for i, (k0, k1) in enumerate(zip(kfs, kfs[1:])):
        print(f"Processing Bundle {i + 1}/{len(kfs) - 1}: Frames {k0}-{k1}")
        graph, vals_rel = setup_ba_graph(k0, k1, use_relative_coords=True)

        if graph.size() < 20:
            print("  -> Skipping bundle, not enough constraints.")
            est_poses.append(est_poses[-1])
            continue

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, vals_rel)
        res_rel = optimizer.optimize()

        motion = res_rel.atPose3(gtsam.symbol('c', k0)).between(res_rel.atPose3(gtsam.symbol('c', k1)))

        if np.linalg.norm(motion.translation()) > 50.0:
            print(f"  -> Rejecting unrealistic motion of {np.linalg.norm(motion.translation()):.1f}m.")
            est_poses.append(est_poses[-1])
            continue

        est_poses.append(est_poses[-1].compose(motion))

    # Plotting
    est_x, est_z = [p.x() for p in est_poses], [p.z() for p in est_poses]
    gt_x, gt_z = [p.x() for p in gt_poses], [p.z() for p in gt_poses]

    plt.figure(figsize=(10, 8))
    plt.plot(est_x, est_z, '-o', label='VO Estimate', markersize=5, lw=2.5, c='cornflowerblue', zorder=10)
    plt.plot(gt_x, gt_z, '--x', label='Ground Truth', markersize=8, lw=2.0, c='darkorange', zorder=5)
    plt.axis('equal');
    plt.legend();
    plt.grid(True);
    plt.title("5.4: VO Trajectory");
    plt.xlabel("X [m]");
    plt.ylabel("Z [m]")
    plt.savefig(OUT_DIR / "trajectory.png");
    plt.close()

    errors = [np.linalg.norm(est_poses[i].translation() - gt_poses[i].translation()) for i in range(len(kfs))]
    plt.figure(figsize=(10, 5))
    plt.plot(kfs, errors, '-o');
    plt.title("Localization Error");
    plt.xlabel("Frame ID");
    plt.ylabel("Error [m]");
    plt.grid(True);
    plt.tight_layout()
    plt.savefig(OUT_DIR / "trajectory_error.png");
    plt.close()

    print("-> Saved final trajectory and error plots.")


def main():
    print("Vision-Aided Navigation: Ex. 5")

    reprojection_and_factor_error()
    analytic_transformation()
    first_ba_window()
    sliding_window_odometry()

    print(f"\nAll tasks completed.")


if __name__ == "__main__":
    main()