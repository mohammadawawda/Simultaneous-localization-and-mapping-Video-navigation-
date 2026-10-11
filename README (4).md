# Visual SLAM on KITTI

Made by Mohammad Awawda.

A full stereo visual SLAM system, run on sequence 05 of the KITTI dataset (2,761 stereo frames from a car's camera).

1. features are matched between the left and right images and triangulated into 3D points
2. the pose of every frame is estimated with PnP inside our own RANSAC
3. every feature is followed across frames in a tracking database (`tracking_database.py`)
4. bundle adjustment with GTSAM on small windows of keyframes
5. a pose graph with loop closures to fix the drift

## Results

Bundle adjustment lowered the mean reprojection error from 2.8 to 0.6 pixels (78.6% less).

The full report with all the plots is in `report.pdf`. It also points to the lines in the code for every part.

## How to run

It needs the KITTI data in `../dataset` (see the main README) and the tracks in `../out_feature_tracking/tracks.pkl`.

```
python project.py
```

The plots are saved in `../out_project/report_figures/`.
