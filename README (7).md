# Stereo Visual SLAM

A stereo visual SLAM system for the KITTI dataset, built step by step. Made by Mohammad Awawda.

Every part builds on the one before it, and `project` puts them all together into the full system.

* [features_and_matching](features_and_matching) AKAZE features and matching between the left and right images, with the ratio test
* [stereo_triangulation](stereo_triangulation) removing bad stereo matches and triangulating 3D points
* [pnp_and_ransac](pnp_and_ransac) the camera movement between two frames with PnP and RANSAC
* [feature_tracking](feature_tracking) following features across all the frames, with statistics and reprojection errors
* [bundle_adjustment](bundle_adjustment) bundle adjustment on windows of keyframes with GTSAM
* [project](project) the full system: PnP with our own RANSAC, bundle adjustment, a pose graph and loop closure

## Data

The KITTI odometry data is not included. Put sequence 05 in a folder named `dataset` here:

* `dataset/sequences/05/` the grayscale images (`image_0`, `image_1`) and `calib.txt`
* `dataset/poses/05.txt` the ground truth poses

## How to run

```
pip install -r requirements.txt
cd feature_tracking
python feature_tracking.py
```

Run every script from inside its own folder. The outputs go to `out_pnp_and_ransac`, `out_feature_tracking`, `out_bundle_adjustment` and `out_project` here.

`bundle_adjustment.py` and the project load the tracks from `out_feature_tracking/tracks.pkl` in the format of `tracking_database.py`. `feature_tracking.py` saves the tracks in a simpler format of its own, so that file has to be made with `TrackingDB`.
