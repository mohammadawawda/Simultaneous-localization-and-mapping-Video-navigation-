# Bundle adjustment

Made by Mohammad Awawda.

The process begins by performing feature detection and matching on stereo image pairs to establish 3D feature tracks across a sequence of frames. These initial feature tracks are used with a Perspective-n-Point (PnP) algorithm to generate a preliminary estimate of the camera's trajectory. This noisy trajectory then serves as the initialization for the core of the project: a sliding-window bundle adjustment (BA) system. Using the GTSAM optimization library, the BA algorithm refines small, overlapping segments of the trajectory by simultaneously optimizing camera poses and 3D landmark positions to minimize reprojection error. 
