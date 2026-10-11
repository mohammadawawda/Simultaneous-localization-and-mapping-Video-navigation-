import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import pickle
from collections import defaultdict
import random

import gtsam

try:
    from tracking_database import TrackingDB
except ImportError:
    print("tracking_database.py not found.")
    sys.exit(1)

# --------------------------------------------------------------------------- #
# PATHS & CONSTANTS
ROOT = Path(__file__).resolve().parent.parent
SEQ_PATH = ROOT / "dataset" / "sequences" / "05"
LEFT_IMG_DIR = SEQ_PATH / "image_0"
RIGHT_IMG_DIR = SEQ_PATH / "image_1"
POSES_FILE = ROOT / "dataset" / "poses" / "05.txt"
CALIB_FILE = SEQ_PATH / "calib.txt"
DB_FILE = ROOT / "out_feature_tracking" / "tracks.pkl"
OUT_DIR = ROOT / "out_project"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = OUT_DIR / "report_figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LANDMARK_KEY_OFFSET = 1_000_000


# --------------------------------------------------------------------------- #
# HELPER FUNCTIONS

def read_kitti_calib(path: Path):
    """Read KITTI calibration file and extract K and baseline."""
    with open(path) as f:
        lines = f.readlines()
    P0 = np.fromstring(lines[0].split(':')[1], sep=' ').reshape(3, 4)
    P1 = np.fromstring(lines[1].split(':')[1], sep=' ').reshape(3, 4)
    K = P0[:, :3]
    baseline = -P1[0, 3] / P1[0, 0]
    return K, baseline


def get_gtsam_calib(K: np.ndarray, baseline: float):
    """Create GTSAM stereo calibration object."""
    return gtsam.Cal3_S2Stereo(K[0, 0], K[1, 1], K[0, 1],
                               K[0, 2], K[1, 2], baseline)


def numpy_to_gtsam_pose(T: np.ndarray) -> gtsam.Pose3:
    """Convert numpy 4x4 matrix to GTSAM Pose3."""
    R = T[:3, :3]
    t = T[:3, 3]
    return gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t[0], t[1], t[2]))


def gtsam_to_numpy_pose(p: gtsam.Pose3) -> np.ndarray:
    """Convert GTSAM Pose3 to numpy 4x4 matrix."""
    T = np.eye(4)
    T[:3, :3] = p.rotation().matrix()
    T[:3, 3] = p.translation()
    return T


def triangulate_point(Pa, Pb, pt_a, pt_b):
    """Triangulate 3D point from two camera projections using SVD.
    Location: slam_project.py, triangulate_point, line 71

    Args:
        Pa: 3x4 projection matrix for camera a
        Pb: 3x4 projection matrix for camera b
        pt_a: 2D point in camera a [x, y]
        pt_b: 2D point in camera b [x, y]

    Returns:
        3D point in homogeneous coordinates
    """
    A = np.vstack([
        pt_a[0] * Pa[2] - Pa[0],
        pt_a[1] * Pa[2] - Pa[1],
        pt_b[0] * Pb[2] - Pb[0],
        pt_b[1] * Pb[2] - Pb[1]
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def compute_relative_error(pose1, pose2, gt_pose1, gt_pose2):
    """Compute relative pose error between estimated and ground truth."""
    # Estimated relative pose
    est_rel = np.linalg.inv(pose1) @ pose2

    # Ground truth relative pose
    gt_rel = np.linalg.inv(gt_pose1) @ gt_pose2

    # Error in relative pose
    error_pose = np.linalg.inv(est_rel) @ gt_rel

    # Extract translation error
    trans_error = np.linalg.norm(error_pose[:3, 3])

    # Extract rotation error (using Rodrigues)
    R_error = error_pose[:3, :3]
    rvec, _ = cv2.Rodrigues(R_error)
    rot_error = np.linalg.norm(rvec) * 180 / np.pi  # in degrees

    return trans_error, rot_error


# --------------------------------------------------------------------------- #
# CUSTOM RANSAC IMPLEMENTATION FOR PnP

def solve_pnp_minimal(object_points, image_points, K):
    """Solve PnP with minimal 3 points using P3P algorithm.

    Args:
        object_points: 3x3 array of 3D points
        image_points: 3x2 array of 2D points
        K: 3x3 camera intrinsic matrix

    Returns:
        (success, rotation_vector, translation_vector)
    """
    # Normalize image points
    normalized_points = np.linalg.inv(K) @ np.vstack([image_points.T, np.ones(3)])
    normalized_points = normalized_points[:2].T

    # Use OpenCV's P3P solver for minimal case
    success, rvecs, tvecs = cv2.solveP3P(
        object_points,
        image_points,
        K,
        None,
        cv2.SOLVEPNP_P3P
    )

    if success and len(rvecs) > 0:
        # Return first solution
        return True, rvecs[0], tvecs[0]
    return False, None, None


def evaluate_pnp_model(object_points, image_points, K, rvec, tvec, threshold=2.0):
    """Evaluate PnP model by computing reprojection errors.

    Returns:
        inliers: boolean array indicating which points are inliers
        errors: array of reprojection errors
    """
    # Project 3D points to image
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, None)
    projected = projected.reshape(-1, 2)

    # Compute reprojection errors
    errors = np.linalg.norm(projected - image_points, axis=1)
    inliers = errors < threshold

    return inliers, errors


def ransac_pnp(object_points, image_points, K, iterations=1000, threshold=2.0, confidence=0.999):
    """Custom RANSAC implementation for PnP.

    """
    n_points = len(object_points)
    if n_points < 3:
        return False, None, None, None

    best_inliers = None
    best_rvec = None
    best_tvec = None
    best_inlier_count = 0

    # Convert to numpy arrays
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)

    for iter_idx in range(iterations):
        # Randomly sample 3 points for P3P
        sample_idx = random.sample(range(n_points), 3)
        sample_obj = object_points[sample_idx]
        sample_img = image_points[sample_idx]

        # Solve minimal problem
        success, rvec, tvec = solve_pnp_minimal(sample_obj, sample_img, K)
        if not success:
            continue

        # Evaluate model on all points
        inlier_mask, errors = evaluate_pnp_model(object_points, image_points, K, rvec, tvec, threshold)
        inlier_count = np.sum(inlier_mask)

        # Update best model if better
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_rvec = rvec
            best_tvec = tvec
            best_inliers = np.where(inlier_mask)[0]

            # Update iteration count based on inlier ratio
            if inlier_count > 3:
                inlier_ratio = inlier_count / n_points
                # Adaptive iteration count - handle edge case where inlier_ratio is 1.0
                if inlier_ratio < 0.99999:  # Avoid divide by zero
                    new_iterations = int(np.log(1 - confidence) / np.log(1 - inlier_ratio ** 3))
                    iterations = min(iterations, new_iterations)
                else:
                    # If almost all points are inliers, we can stop early
                    break

        # Early termination if enough inliers found
        if best_inlier_count > 0.8 * n_points:
            break

    if best_inlier_count < 6:  # Need at least 6 inliers for reliable pose
        return False, None, None, None

    # Refine pose using all inliers
    inlier_obj = object_points[best_inliers]
    inlier_img = image_points[best_inliers]

    # Use iterative PnP for refinement
    success, refined_rvec, refined_tvec = cv2.solvePnP(
        inlier_obj, inlier_img, K, None,
        rvec=best_rvec, tvec=best_tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        return True, refined_rvec, refined_tvec, best_inliers
    else:
        return True, best_rvec, best_tvec, best_inliers


# --------------------------------------------------------------------------- #
# TRACKING STATISTICS

def compute_tracking_statistics(db: TrackingDB):
    """Compute and visualize tracking statistics.
    """
    print("\n=== Computing Tracking Statistics ===")

    stats = {
        'total_tracks': db.track_num(),
        'total_frames': db.frame_num(),
        'track_lengths': [],
        'frame_links': [],
        'matches_per_frame': [],
        'inliers_per_frame': []
    }

    # Compute track lengths
    for tid in range(db.track_num()):
        track_frames = db.frames(tid)
        if track_frames:
            stats['track_lengths'].append(len(track_frames))

    # Compute frame statistics
    for fid in range(db.frame_num()):
        links = db.all_frame_links(fid)
        stats['frame_links'].append(len(links))

        # For matches and inliers, we need to check with next frame
        if fid < db.frame_num() - 1:
            tracks_current = set(db.tracks(fid))
            tracks_next = set(db.tracks(fid + 1))
            matches = len(tracks_current.intersection(tracks_next))
            stats['matches_per_frame'].append(matches)

            # Assume all continuing tracks are inliers
            if len(tracks_current) > 0:
                inlier_ratio = matches / len(tracks_current)
                stats['inliers_per_frame'].append(inlier_ratio)

    # Print statistics
    print(f"Total number of tracks: {stats['total_tracks']}")
    print(f"Number of frames: {stats['total_frames']}")
    print(f"Mean track length: {np.mean(stats['track_lengths']):.2f}")
    print(f"Mean number of frame links: {np.mean(stats['frame_links']):.2f}")

    # Plot statistics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Number of matches per frame
    ax = axes[0, 0]
    ax.plot(stats['matches_per_frame'])
    ax.set_xlabel('Frame')
    ax.set_ylabel('Number of matches')
    ax.set_title('Matches per Frame')
    ax.grid(True)

    # 2. Percentage of inliers per frame
    ax = axes[0, 1]
    ax.plot(np.array(stats['inliers_per_frame']) * 100)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Inlier percentage (%)')
    ax.set_title('Inlier Percentage per Frame')
    ax.grid(True)

    # 3. Connectivity graph
    ax = axes[1, 0]
    connectivity = []
    for fid in range(db.frame_num() - 1):
        tracks_current = set(db.tracks(fid))
        tracks_next = set(db.tracks(fid + 1))
        outgoing = len(tracks_current.intersection(tracks_next))
        connectivity.append(outgoing)
    ax.plot(connectivity)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Outgoing tracks')
    ax.set_title('Connectivity: Outgoing Tracks to Next Frame')
    ax.grid(True)

    # 4. Track length histogram (log scale)
    ax = axes[1, 1]
    ax.hist(stats['track_lengths'], bins=50, edgecolor='black')
    ax.set_xlabel('Track length')
    ax.set_ylabel('Count')
    ax.set_title('Track Length Histogram')
    ax.set_yscale('log')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'tracking_statistics.png', dpi=150)
    plt.close()
    print(f"Figure saved: tracking_statistics.png ")

    return stats


# --------------------------------------------------------------------------- #
# INITIAL ODOMETRY (PnP) WITH RANSAC

def initial_odometry_poses(db: TrackingDB, K: np.ndarray, baseline: float):
    """Compute initial visual odometry trajectory using PnP with RANSAC.

    This function implements PnP trajectory calculation using our custom RANSAC.
    """
    print("\nComputing initial visual odometry trajectory (PnP with RANSAC)...")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    camera_poses = [np.eye(4)]  # First camera at origin
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([np.eye(3), np.array([[-baseline, 0, 0]]).T])

    # For tracking projection errors
    pnp_projection_errors = defaultdict(list)

    # Store RANSAC statistics
    ransac_stats = {
        'inlier_counts': [],
        'total_matches': []
    }

    for i in tqdm(range(1, db.frame_num()), desc="PnP"):
        links_prev = db.all_frame_links(i - 1)
        des_prev = db.features(i - 1)
        des_cur = db.features(i)

        matches = bf.knnMatch(des_prev, des_cur, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        obj_pts_cam = []
        img_pts_cur = []
        track_ids = []

        for m in good:
            link_prev = links_prev[m.queryIdx]
            # Triangulate 3D point from stereo in previous frame
            pt3d_cam = triangulate_point(P0, P1, link_prev.left_keypoint(), link_prev.right_keypoint())

            if 0.5 < pt3d_cam[2] < 30:  # Depth check
                obj_pts_cam.append(pt3d_cam)
                link_cur = db.all_frame_links(i)[m.trainIdx]
                img_pts_cur.append([link_cur.x_left, link_cur.y])

                # Track which track this corresponds to
                prev_tracks = db.tracks(i - 1)
                if m.queryIdx < len(prev_tracks):
                    track_ids.append(prev_tracks[m.queryIdx])

        if len(obj_pts_cam) < 6:  # Need minimum points for RANSAC
            camera_poses.append(camera_poses[-1])
            ransac_stats['inlier_counts'].append(0)
            ransac_stats['total_matches'].append(len(good))
            continue

        obj_pts_cam = np.array(obj_pts_cam, dtype=np.float32)
        img_pts_cur = np.array(img_pts_cur, dtype=np.float32)

        # Use custom RANSAC PnP
        success, rvec, tvec, inliers = ransac_pnp(
            obj_pts_cam, img_pts_cur, K,
            iterations=1000,
            threshold=1.5,
            confidence=0.999
        )

        if not success or inliers is None or len(inliers) < 10:
            camera_poses.append(camera_poses[-1])
            ransac_stats['inlier_counts'].append(0)
            ransac_stats['total_matches'].append(len(good))
            continue

        # Store RANSAC statistics
        ransac_stats['inlier_counts'].append(len(inliers))
        ransac_stats['total_matches'].append(len(obj_pts_cam))

        # Compute projection errors for inlier tracks
        R_rel, _ = cv2.Rodrigues(rvec)
        for idx in inliers:
            if idx < len(track_ids):
                # Project point and compute error
                pt_3d = obj_pts_cam[idx]
                pt_proj = K @ (R_rel @ pt_3d + tvec.flatten())
                pt_proj = pt_proj[:2] / pt_proj[2]
                error = np.linalg.norm(pt_proj - img_pts_cur[idx])

                tid = track_ids[idx]
                frame_dist = 1  # Distance from triangulation frame
                pnp_projection_errors[tid].append((frame_dist, error))

        # Build transformation (camera pose in world frame)
        T_curr_to_prev = np.eye(4)
        T_curr_to_prev[:3, :3] = R_rel
        T_curr_to_prev[:3, 3] = tvec.flatten()

        T_world_to_prev = camera_poses[-1]
        T_prev_to_curr = np.linalg.inv(T_curr_to_prev)
        T_world_to_curr = T_world_to_prev @ T_prev_to_curr

        camera_poses.append(T_world_to_curr)

    # Convert to GTSAM format
    gtsam_poses = [numpy_to_gtsam_pose(T) for T in camera_poses]

    # Plot RANSAC performance
    if ransac_stats['inlier_counts']:
        plt.figure(figsize=(10, 5))
        inlier_ratios = [ic / tm if tm > 0 else 0
                         for ic, tm in zip(ransac_stats['inlier_counts'],
                                           ransac_stats['total_matches'])]
        plt.plot(inlier_ratios, 'b-', linewidth=1)
        plt.xlabel('Frame')
        plt.ylabel('RANSAC Inlier Ratio')
        plt.title('PnP RANSAC Performance')
        plt.grid(True)
        plt.savefig(REPORT_DIR / 'pnp_ransac_performance.png')
        plt.close()
        print(f"Figure saved: pnp_ransac_performance.png ")

    return gtsam_poses, pnp_projection_errors


# --------------------------------------------------------------------------- #
# BUNDLE ADJUSTMENT

def create_single_bundle(db: TrackingDB, frames: list, init_poses: dict,
                         K_gtsam, world_to_local_transform):
    """Create and optimize a single bundle.

    """
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Noise models
    meas_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]))
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]))

    # Add prior on first frame (at origin in bundle coordinates)
    graph.add(gtsam.PriorFactorPose3(frames[0], gtsam.Pose3(), prior_noise))
    initial.insert(frames[0], gtsam.Pose3())

    # Initialize all poses in bundle (in local coordinates)
    for i, fid in enumerate(frames[1:], 1):
        if fid < len(init_poses):
            world_pose = init_poses[fid]
            local_pose = world_to_local_transform.between(world_pose)
            initial.insert(fid, local_pose)

            # Add odometry factor between consecutive poses in the bundle
            # Use the previous frame in the frames list, not fid-1
            prev_fid = frames[i - 1]
            if initial.exists(prev_fid) and initial.exists(fid):
                prev_pose = initial.atPose3(prev_fid)
                curr_pose = initial.atPose3(fid)
                relative = prev_pose.between(curr_pose)
                graph.add(gtsam.BetweenFactorPose3(prev_fid, fid, relative, odom_noise))

    # Collect tracks visible in this bundle
    tracks = set()
    for fid in frames:
        if fid < db.frame_num():  # Check bounds
            tracks.update(db.tracks(fid))
    tracks = list(tracks)[:200]  # Limit for computational efficiency

    # Add landmarks and measurement factors
    added_landmarks = 0
    for tid in tracks:
        visible_frames = [f for f in frames if f < db.frame_num() and tid in db.tracks(f)]
        if len(visible_frames) < 2:
            continue

        landmark_key = tid + LANDMARK_KEY_OFFSET

        try:
            # Triangulate from first observation
            first_frame = visible_frames[0]
            if not initial.exists(first_frame):
                continue

            link = db.link(first_frame, tid)
            stereo_pt = gtsam.StereoPoint2(link.x_left, link.x_right, link.y)
            cam = gtsam.StereoCamera(initial.atPose3(first_frame), K_gtsam)
            pt_3d = cam.backproject(stereo_pt)

            # Depth check
            if pt_3d[2] < 0.5 or pt_3d[2] > 50:
                continue

            initial.insert(landmark_key, pt_3d)
            added_landmarks += 1

            # Add measurement factors for all observations
            for fid in visible_frames:
                if initial.exists(fid):  # Only add factor if pose exists
                    link = db.link(fid, tid)
                    meas = gtsam.StereoPoint2(link.x_left, link.x_right, link.y)
                    factor = gtsam.GenericStereoFactor3D(meas, meas_noise, fid, landmark_key, K_gtsam)
                    graph.add(factor)

        except Exception as e:
            continue

    # Optimize if we have factors
    if graph.size() > 0:
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity("SILENT")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()

        # Extract covariances for poses
        try:
            marginals = gtsam.Marginals(graph, result)
            covariances = {}
            for fid in frames:
                if result.exists(fid):
                    covariances[fid] = marginals.marginalCovariance(fid)
            return graph, initial, result, covariances
        except:
            return graph, initial, result, {}
    else:
        return graph, initial, initial, {}


def extract_relative_transformation_with_covariance(bundle_result, from_frame, to_frame):
    """Extract relative transformation and covariance from bundle result.

    Args:
        bundle_result: Optimization result from bundle adjustment
        from_frame: Source frame index
        to_frame: Target frame index

    Returns:
        (relative_pose, covariance_matrix)
    """
    if bundle_result.exists(from_frame) and bundle_result.exists(to_frame):
        pose_from = bundle_result.atPose3(from_frame)
        pose_to = bundle_result.atPose3(to_frame)
        relative = pose_from.between(pose_to)

        # For covariance, we approximate it based on the optimization result
        # In practice, this would be extracted from the marginals
        # Here we use a default covariance
        covariance = np.eye(6)
        covariance[:3, :3] *= 0.01  # Rotation uncertainty
        covariance[3:, 3:] *= 0.1  # Translation uncertainty

        return relative, covariance
    return None, None


def solve_bundle_adjustment(db: TrackingDB, init_poses: list, K_gtsam, gt_poses_np: list):
    """Bundle adjustment with sliding windows."""
    print("\n--- Solving Bundle Adjustment ---")

    # Define keyframes
    bundle_size = 10
    max_frame = min(500, db.frame_num())  # Process first 500 frames
    keyframes = list(range(0, max_frame, bundle_size))
    if keyframes[-1] != max_frame - 1:
        keyframes.append(max_frame - 1)

    print(f"Processing {len(keyframes)} keyframes up to frame {max_frame}")

    # Store results
    bundle_results = {
        'optimized_poses': {0: init_poses[0]},
        'bundle_errors': [],
        'projection_errors': defaultdict(list),
        'relative_poses': [],
        'covariances': {}
    }

    # Process each bundle window
    for bundle_idx in tqdm(range(len(keyframes) - 1), desc="Bundles"):
        kf_start = keyframes[bundle_idx]
        kf_end = keyframes[bundle_idx + 1]
        frames = list(range(kf_start, min(kf_end + 1, len(init_poses), max_frame)))

        # Transform to local bundle coordinates
        world_to_local = bundle_results['optimized_poses'][kf_start]

        # Create single bundle
        graph, initial, result, covariances = create_single_bundle(
            db, frames, init_poses, K_gtsam, world_to_local
        )

        if graph.size() > 0:
            initial_error = graph.error(initial)
            final_error = graph.error(result)

            # Store bundle statistics
            bundle_results['bundle_errors'].append((
                initial_error / graph.size() if graph.size() > 0 else 0,
                final_error / graph.size() if graph.size() > 0 else 0,
                graph.size()
            ))

            # Convert optimized poses back to world coordinates
            for fid in frames:
                if result.exists(fid):
                    local_opt = result.atPose3(fid)
                    world_opt = world_to_local.compose(local_opt)
                    bundle_results['optimized_poses'][fid] = world_opt

            # Extract relative transformation with covariance
            if len(frames) > 1:
                rel_pose, cov = extract_relative_transformation_with_covariance(
                    result, frames[0], frames[-1]
                )
                if rel_pose is not None:
                    bundle_results['relative_poses'].append({
                        'from': frames[0],
                        'to': frames[-1],
                        'transform': rel_pose,
                        'covariance': cov
                    })

            print(f"  Bundle {bundle_idx}: {graph.size()} factors, "
                  f"error {initial_error:.1f} -> {final_error:.1f}")
        else:
            # No factors, propagate poses
            for fid in frames[1:]:
                if fid < len(init_poses):
                    bundle_results['optimized_poses'][fid] = init_poses[fid]
            bundle_results['bundle_errors'].append((0, 0, 0))

    return bundle_results, keyframes

# --------------------------------------------------------------------------- #
# POSE GRAPH BUILDING AND LOOP CLOSURE

def build_pose_graph(bundle_results, keyframes):
    """Build pose graph from bundle adjustment results.

    This function builds the pose graph structure from bundle adjustment results.
    """
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Prior on first pose
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]))
    first_pose = bundle_results['optimized_poses'][keyframes[0]]
    graph.add(gtsam.PriorFactorPose3(keyframes[0], first_pose, prior_noise))

    # Add all keyframe poses
    for kf in keyframes:
        if kf in bundle_results['optimized_poses']:
            initial.insert(kf, bundle_results['optimized_poses'][kf])

    # Add relative pose constraints from bundle adjustment
    for rel_data in bundle_results['relative_poses']:
        # Create noise model from covariance
        if isinstance(rel_data['covariance'], np.ndarray):
            cov_matrix = rel_data['covariance'] + np.eye(6) * 1e-6
            noise = gtsam.noiseModel.Gaussian.Covariance(cov_matrix)
        else:
            noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]))

        factor = gtsam.BetweenFactorPose3(
            rel_data['from'],
            rel_data['to'],
            rel_data['transform'],
            noise
        )
        graph.add(factor)

    return graph, initial


def add_loop_closure_cheat(graph, initial, keyframes, gt_poses_np):
    """Add loop closure using ground truth cheat for last frame.

    This implements the loop closure detection and factor creation using
    the ground truth "cheat" approach for the last frame.
    """
    last_kf = keyframes[-1]
    if last_kf < len(gt_poses_np):
        gt_last_pose = numpy_to_gtsam_pose(gt_poses_np[last_kf])

        # Get current estimate
        current_last = initial.atPose3(last_kf)
        gt_position = gt_last_pose.translation()
        current_position = current_last.translation()

        print(f"\nLoop closure cheat on frame {last_kf}:")
        print(f"  Current estimate: [{current_position[0]:.1f}, {current_position[1]:.1f}, {current_position[2]:.1f}]")
        print(f"  Ground truth: [{gt_position[0]:.1f}, {gt_position[1]:.1f}, {gt_position[2]:.1f}]")
        print(f"  Error: {np.linalg.norm(current_position - gt_position):.1f} m")

        # Add loop closure factor with tight noise
        loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.001, 0.001, 0.001]))
        loop_factor = gtsam.PriorFactorPose3(last_kf, gt_last_pose, loop_noise)
        graph.add(loop_factor)

        return True
    return False


def create_pose_graph_with_loop_closure(bundle_results, keyframes, gt_poses_np):
    """Create and optimize pose graph with loop closure using ground truth cheat approach."""
    print("\n--- Creating Pose Graph with Loop Closure (Cheat) ---")

    # Build pose graph without loop closure
    graph_no_lc, initial = build_pose_graph(bundle_results, keyframes)

    # Copy for version with loop closure
    graph_with_lc = gtsam.NonlinearFactorGraph()
    for i in range(graph_no_lc.size()):
        graph_with_lc.add(graph_no_lc.at(i))

    # Optimize WITHOUT loop closure
    print("Optimizing pose graph without loop closure...")
    optimizer_no_lc = gtsam.LevenbergMarquardtOptimizer(graph_no_lc, initial)
    result_no_lc = optimizer_no_lc.optimize()
    error_no_lc = graph_no_lc.error(result_no_lc)
    print(f"Pose graph error without LC: {error_no_lc:.2f}")

    # Add loop closure cheat
    if add_loop_closure_cheat(graph_with_lc, result_no_lc, keyframes, gt_poses_np):
        # Optimize WITH loop closure
        print("Optimizing pose graph with loop closure...")
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(100)
        optimizer_with_lc = gtsam.LevenbergMarquardtOptimizer(graph_with_lc, result_no_lc, params)
        result_with_lc = optimizer_with_lc.optimize()
        error_with_lc = graph_with_lc.error(result_with_lc)
        print(f"Pose graph error with LC: {error_with_lc:.2f}")
    else:
        result_with_lc = result_no_lc

    # Extract covariances
    try:
        marginals_no_lc = gtsam.Marginals(graph_no_lc, result_no_lc)
        marginals_with_lc = gtsam.Marginals(graph_with_lc, result_with_lc)

        covariances_no_lc = {}
        covariances_with_lc = {}

        for kf in keyframes:
            if result_no_lc.exists(kf):
                covariances_no_lc[kf] = marginals_no_lc.marginalCovariance(kf)
            if result_with_lc.exists(kf):
                covariances_with_lc[kf] = marginals_with_lc.marginalCovariance(kf)
    except:
        covariances_no_lc = {}
        covariances_with_lc = {}

    return result_no_lc, result_with_lc, covariances_no_lc, covariances_with_lc
def plot_trajectories_comparison(init_poses, bundle_results, pose_graph_result,
                                 keyframes, gt_poses_np, out_dir):
    """Plot trajectory comparison.
    """
    plt.figure(figsize=(14, 10))

    # Ground truth
    gt_positions = np.array([T[:3, 3] for T in gt_poses_np])
    plt.plot(gt_positions[:, 0], gt_positions[:, 2], 'r-',
             label='Ground Truth', linewidth=2)

    # Initial PnP (limit to processed frames)
    pnp_positions = []
    for i, pose in enumerate(init_poses):
        if i < len(gt_poses_np):
            T = gtsam_to_numpy_pose(pose)
            pnp_positions.append(T[:3, 3])
    pnp_positions = np.array(pnp_positions)
    plt.plot(pnp_positions[:, 0], pnp_positions[:, 2], 'g--',
             label='PnP Estimation', linewidth=1.5, alpha=0.7)

    # Bundle adjustment keyframes
    ba_positions = []
    for kf in keyframes:
        if kf in bundle_results['optimized_poses']:
            pose = bundle_results['optimized_poses'][kf]
            ba_positions.append(pose.translation())
    if ba_positions:
        ba_positions = np.array(ba_positions)
        plt.plot(ba_positions[:, 0], ba_positions[:, 2], 'b.-',
                 label='Bundle Adjustment', markersize=6)

    # Pose graph with loop closure
    if pose_graph_result is not None:
        pg_positions = []
        for kf in keyframes:
            if pose_graph_result.exists(kf):
                pose = pose_graph_result.atPose3(kf)
                pg_positions.append(pose.translation())
        if pg_positions:
            pg_positions = np.array(pg_positions)
            plt.plot(pg_positions[:, 0], pg_positions[:, 2], 'm.-',
                     label='Pose Graph (with LC)', markersize=6)

    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.title('Trajectory Comparison: PnP vs Bundle vs Pose Graph')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(out_dir / 'trajectory_comparison.png', dpi=150)
    plt.close()


def plot_optimization_errors(bundle_results, keyframes, out_dir):
    """Plot bundle optimization errors.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Mean factor error
    bundle_indices = []
    initial_errors = []
    final_errors = []

    for i, (init_err, final_err, n_factors) in enumerate(bundle_results['bundle_errors']):
        if n_factors > 0 and i < len(keyframes) - 1:
            bundle_indices.append(keyframes[i])
            initial_errors.append(init_err)
            final_errors.append(final_err)

    if bundle_indices:
        ax1.plot(bundle_indices, initial_errors, 'r-', label='Initial Error', linewidth=2)
        ax1.plot(bundle_indices, final_errors, 'b-', label='Optimized Error', linewidth=2)
        ax1.set_xlabel('Bundle Starting at Frame ID')
        ax1.set_ylabel('Mean Factor Error')
        ax1.set_title('Mean Factor Error Before and After Bundle Optimization')
        ax1.legend()
        ax1.grid(True)
        ax1.set_yscale('log')

    # Median projection error
    median_before = []
    median_after = []
    valid_bundles = []

    for i, kf in enumerate(keyframes[:-1]):
        if i in bundle_results['projection_errors']:
            errors = bundle_results['projection_errors'][i]
            if errors['before'] and errors['after']:
                median_before.append(np.median(errors['before']))
                median_after.append(np.median(errors['after']))
                valid_bundles.append(kf)

    if valid_bundles:
        ax2.plot(valid_bundles, median_before, 'r-', label='Before Optimization', linewidth=2)
        ax2.plot(valid_bundles, median_after, 'b-', label='After Optimization', linewidth=2)
        ax2.set_xlabel('Bundle Starting at Frame ID')
        ax2.set_ylabel('Median Projection Error [pixels]')
        ax2.set_title('Median Projection Error Before and After Bundle Optimization')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / 'optimization_errors.png', dpi=150)
    plt.close()


def compute_absolute_errors(poses, gt_poses_np, name, out_dir):
    """Compute and plot absolute pose errors.
    """
    errors = {
        'x': [], 'y': [], 'z': [], 'norm': [], 'angle': []
    }

    frames = []

    for i, pose in enumerate(poses):
        if i >= len(gt_poses_np):
            break

        # Convert pose to numpy
        if hasattr(pose, 'matrix'):  # GTSAM pose
            T_est = gtsam_to_numpy_pose(pose)
        else:  # Already numpy
            T_est = pose

        T_gt = gt_poses_np[i]

        # Position error
        pos_error = T_est[:3, 3] - T_gt[:3, 3]
        errors['x'].append(abs(pos_error[0]))
        errors['y'].append(abs(pos_error[1]))
        errors['z'].append(abs(pos_error[2]))
        errors['norm'].append(np.linalg.norm(pos_error))

        # Angle error
        R_error = T_est[:3, :3].T @ T_gt[:3, :3]
        rvec, _ = cv2.Rodrigues(R_error)
        angle_error = np.linalg.norm(rvec) * 180 / np.pi
        errors['angle'].append(angle_error)

        frames.append(i)

    if not frames:
        return errors

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(frames, errors['x'], 'r-', label='X error', alpha=0.7)
    ax1.plot(frames, errors['y'], 'g-', label='Y error', alpha=0.7)
    ax1.plot(frames, errors['z'], 'b-', label='Z error', alpha=0.7)
    ax1.plot(frames, errors['norm'], 'k-', label='Total error', linewidth=2)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Position Error [m]')
    ax1.set_title(f'{name} - Absolute Position Error')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(frames, errors['angle'], 'm-', linewidth=2)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Angle Error [degrees]')
    ax2.set_title(f'{name} - Absolute Angle Error')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / f'absolute_error_{name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()

    return errors


def compute_relative_errors_over_distances(poses, gt_poses_np, distances, name, out_dir):
    """Compute relative errors over different sequence lengths.
    """
    results = {}

    for dist in distances:
        trans_errors = []
        rot_errors = []

        for start_idx in range(0, len(poses) - dist):
            end_idx = start_idx + dist

            if end_idx >= len(poses) or end_idx >= len(gt_poses_np):
                break

            # Get poses
            if hasattr(poses[start_idx], 'matrix'):  # GTSAM
                T_start = gtsam_to_numpy_pose(poses[start_idx])
                T_end = gtsam_to_numpy_pose(poses[end_idx])
            else:
                T_start = poses[start_idx]
                T_end = poses[end_idx]

            # Compute ground truth distance traveled
            gt_dist = 0
            for i in range(start_idx, end_idx):
                if i + 1 < len(gt_poses_np):
                    gt_dist += np.linalg.norm(gt_poses_np[i + 1][:3, 3] - gt_poses_np[i][:3, 3])

            if gt_dist > 0:
                trans_err, rot_err = compute_relative_error(T_start, T_end,
                                                            gt_poses_np[start_idx],
                                                            gt_poses_np[end_idx])

                # Normalize by distance
                trans_errors.append(100 * trans_err / gt_dist)  # Percentage
                rot_errors.append(rot_err / gt_dist)  # deg/m

        results[dist] = {
            'trans_errors': trans_errors,
            'rot_errors': rot_errors,
            'mean_trans': np.mean(trans_errors) if trans_errors else 0,
            'mean_rot': np.mean(rot_errors) if rot_errors else 0
        }

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    for dist in distances:
        if results[dist]['trans_errors']:
            ax1.plot(results[dist]['trans_errors'], label=f'Seq length {dist}', alpha=0.7)

    ax1.set_xlabel('Sequence Index')
    ax1.set_ylabel('Translation Error [%]')
    ax1.set_title(f'{name} - Relative Translation Error')
    ax1.legend()
    ax1.grid(True)

    for dist in distances:
        if results[dist]['rot_errors']:
            ax2.plot(results[dist]['rot_errors'], label=f'Seq length {dist}', alpha=0.7)

    ax2.set_xlabel('Sequence Index')
    ax2.set_ylabel('Rotation Error [deg/m]')
    ax2.set_title(f'{name} - Relative Rotation Error')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / f'relative_error_{name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()

    # Print averages
    print(f"\n{name} - Average Relative Errors:")
    for dist in distances:
        print(f"  Sequence length {dist}:")
        print(f"    Translation: {results[dist]['mean_trans']:.2f}%")
        print(f"    Rotation: {results[dist]['mean_rot']:.3f} deg/m")

    return results


def plot_uncertainty_evolution(covariances, keyframes, out_dir):
    """Plot uncertainty evolution from pose graph.
    """
    location_uncertainty = []
    angle_uncertainty = []
    valid_keyframes = []

    for kf in keyframes:
        if kf in covariances:
            cov = covariances[kf]

            # Location uncertainty (trace of position part)
            loc_uncertainty = np.sqrt(np.trace(cov[3:6, 3:6]))
            location_uncertainty.append(loc_uncertainty)

            # Angle uncertainty (trace of rotation part)
            ang_uncertainty = np.sqrt(np.trace(cov[0:3, 0:3])) * 180 / np.pi
            angle_uncertainty.append(ang_uncertainty)

            valid_keyframes.append(kf)

    if not valid_keyframes:
        return

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(valid_keyframes, location_uncertainty, 'b-', linewidth=2)
    plt.xlabel('Keyframe Index')
    plt.ylabel('Location Uncertainty [m]')
    plt.title('Location Uncertainty vs Keyframe (with Loop Closure)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(valid_keyframes, angle_uncertainty, 'r-', linewidth=2)
    plt.xlabel('Keyframe Index')
    plt.ylabel('Angle Uncertainty [degrees]')
    plt.title('Angle Uncertainty vs Keyframe (with Loop Closure)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / 'uncertainty_with_loop_closure.png', dpi=150)
    plt.close()


def plot_projection_error_vs_distance(projection_errors, name, out_dir):
    """Plot projection error as function of distance from reference frame.
    """
    # Aggregate errors by distance
    distance_errors = defaultdict(list)

    for track_id, errors in projection_errors.items():
        for dist, error in errors:
            distance_errors[dist].append(error)

    # Compute statistics
    distances = sorted(distance_errors.keys())
    medians = []
    percentile_25 = []
    percentile_75 = []

    for dist in distances:
        if distance_errors[dist]:
            medians.append(np.median(distance_errors[dist]))
            percentile_25.append(np.percentile(distance_errors[dist], 25))
            percentile_75.append(np.percentile(distance_errors[dist], 75))

    if not distances or not medians:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(distances[:len(medians)], medians, 'b-', linewidth=2, label='Median')
    plt.fill_between(distances[:len(medians)], percentile_25, percentile_75,
                     alpha=0.3, label='25-75 percentile')
    plt.xlabel('Distance from Reference Frame')
    plt.ylabel('Projection Error [pixels]')
    plt.title(f'{name} - Projection Error vs Distance from Reference Frame')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f'projection_error_distance_{name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()

def main():
    """Main function to run complete SLAM pipeline with analysis."""

    # Load tracking database
    if not DB_FILE.exists():
        print(f"Error: Tracking database not found at {DB_FILE}")
        return

    db = TrackingDB()
    db.load(str(DB_FILE).replace('.pkl', ''))
    print(f"Loaded tracking database: {db.frame_num()} frames, {db.track_num()} tracks")

    # Load calibration
    K, baseline = read_kitti_calib(CALIB_FILE)
    K_gtsam = get_gtsam_calib(K, baseline)
    print(f"Loaded calibration: baseline = {baseline:.3f} m")

    # Load ground truth
    gt_poses_np = []
    with open(POSES_FILE) as f:
        for line in f:
            values = np.array([float(x) for x in line.split()])
            if len(values) == 12:
                T = np.eye(4)
                T[:3, :] = values.reshape(3, 4)
                gt_poses_np.append(T)
    print(f"Loaded {len(gt_poses_np)} ground truth poses")

    # 1. Compute tracking statistics
    tracking_stats = compute_tracking_statistics(db)

    # 2. Initial PnP odometry
    init_poses, pnp_projection_errors = initial_odometry_poses(db, K, baseline)

    # 3. Bundle adjustment
    bundle_results, keyframes = solve_bundle_adjustment(db, init_poses, K_gtsam, gt_poses_np)

    # 4. Pose graph with loop closure
    result_no_lc, result_with_lc, covariances_no_lc, covariances_with_lc = create_pose_graph_with_loop_closure(
        bundle_results, keyframes, gt_poses_np)
    # 5. Generate all visualizations and analyses
    print("\n=== Generating Report Figures ===")

    # Trajectory comparison
    plot_trajectories_comparison(init_poses, bundle_results, result_with_lc,
                                 keyframes, gt_poses_np, REPORT_DIR)

    # Optimization errors
    plot_optimization_errors(bundle_results, keyframes, REPORT_DIR)

    # Absolute errors
    print("\nComputing absolute errors...")
    pnp_abs_errors = compute_absolute_errors(init_poses, gt_poses_np, "PnP Estimation", REPORT_DIR)

    # For bundle adjustment, we need to interpolate keyframes
    ba_poses_full = []
    for i in range(len(init_poses)):
        # Find closest keyframe
        closest_kf = min(keyframes, key=lambda x: abs(x - i))
        if closest_kf in bundle_results['optimized_poses']:
            ba_poses_full.append(bundle_results['optimized_poses'][closest_kf])
        else:
            ba_poses_full.append(init_poses[i])

    ba_abs_errors = compute_absolute_errors(ba_poses_full, gt_poses_np,
                                            "Bundle Adjustment", REPORT_DIR)

    # Relative errors over distances
    print("\nComputing relative errors...")
    distances = [100, 400, 800]
    pnp_rel_errors = compute_relative_errors_over_distances(
        init_poses, gt_poses_np, distances, "PnP", REPORT_DIR)

    # For bundle adjustment, use keyframes only
    ba_poses_kf = [bundle_results['optimized_poses'][kf] for kf in keyframes
                   if kf in bundle_results['optimized_poses']]
    ba_rel_errors = compute_relative_errors_over_distances(
        ba_poses_kf, gt_poses_np, [10, 40, 80], "Bundle Adjustment", REPORT_DIR)

    # Uncertainty evolution
    plot_uncertainty_evolution(covariances_with_lc, keyframes, REPORT_DIR)

    # Projection error vs distance
    plot_projection_error_vs_distance(pnp_projection_errors, "PnP", REPORT_DIR)

    # Summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total frames processed: {db.frame_num()}")
    print(f"Total tracks: {db.track_num()}")
    print(f"Number of keyframes: {len(keyframes)}")
    print(f"Average track length: {np.mean(tracking_stats['track_lengths']):.2f}")

    # Save results for later analysis
    results = {
        'tracking_stats': tracking_stats,
        'keyframes': keyframes,
        'pnp_abs_errors': pnp_abs_errors,
        'ba_abs_errors': ba_abs_errors,
        'pnp_rel_errors': pnp_rel_errors,
        'ba_rel_errors': ba_rel_errors
    }

    with open(REPORT_DIR / 'analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\nAll results saved to {REPORT_DIR}")
    print("\n✓ Complete SLAM pipeline with analysis finished!")


if __name__ == "__main__":
    main()