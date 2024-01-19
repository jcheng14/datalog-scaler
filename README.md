# Datalog Log Scaler

This code repo hosts the python scripts for developing the log scaling algorithms to process and analyze the point cloud data (PCD) of a stacked log pile. It is part of the Fresh/Datalog MVP Phase 1 data processing pipeline workstream to obtain the log count, diameter, depth/length and log-end pair matching information using a machine vision approach.

## Technical Approaches

Two technical approaches are used to generate the PCD:

* Indirect approach: Extract log pile PCD from a mesh that is reconstructed by using photogrammetry on a group of spatially overlapped RGB images of the log pile.

  * [iPhone 14 Pro Max back camera](https://www.apple.com/iphone-14-pro/), 12MP & 48MP modes
  * [Canon EOS 90D DLSR camera](https://www.usa.canon.com/shop/p/eos-90d?color=Black&type=New) with [an optical lens](url to be confirmed later), 32.5MP
  * [Flir Blackfly-PGE-200S6C machine vision camera](https://www.edmundoptics.com/p/bfs-pge-200s6c-c-poe-gige-blackflyr-s-color-camera/40196/), 20MP with an [8-mm lens](https://www.edmundoptics.com/p/8mm-focal-length-hp-series-fixed-focal-length-lens/41693/)

* Direct approach: Acquire log pile PCD from a lidar.

  * [Livox Mid-70](https://www.livoxtech.com/mid-70)
  * [Livox HAP(TX)](https://www.livoxtech.com/hap)
  * Refer to the relevant [code repo](https://bitbucket.org/freshconsulting/datalog-mvp/src/main/) for the data acquisition of the direct approach

## Relevant Software Tools

* [Meshroom](https://alicevision.org/#meshroom) is the choosen tool for the photogrammetry mesh reconstruction work
* [Open3D](http://www.open3d.org/) is used for PCD manipulation, analysis, and visualization
* [Meshlab](https://www.meshlab.net/) is used for mesh visualization and measurement, analysis with a GUI

## Log Scaling Processing Pipeline

### Load the log PCD (using ```log_loading.py```)

1. Loading mesh/pcd from the raw PCD dataset that was obtained from either photogrammetry or Lidar scan
2. Volume of interest crop to narrow down to the relevant PCD portion (needs manual selection)

### Scale a single log-end PCD (use ```log_scaling.py```)

1. Load the ```xxxx_crop.PLY``` PCD dataset from the log PCD loading step, and reject outlier points to denoise/sparse the PCD
2. Pre-processiong the PCD from the previous step by detecting a ground plane out of the PCD, and aligning the PCD ground plane normal with one of the global xyz-axes per user's spec
3. Define three parameters for the scaling process: planar patch candidate detection and segmentation, fiducial detection for selecting the two vertical stanchion poles closet toward the sensor plane, log-end selection
4. Log-end scaling process based on the PCD obtained in step 2:
    * Reconfirm the ground plane
    * Detect planar patches (point-density and planar-cluster based segmentation)
    * Find a reference plane from the detected two vertical stanchion poles cloest toward the sensor plane
    * Filter the detected planar patches w.r.t. the reference plane normal vector and patch rectangle aspect ratio criteria
    * Obtain patch side length and width (rough estimation of log-end diameters), as well as the depth to the reference plane (log-end depth), from the filtered planar patches
    * Visualization and report generation

### Scale both log-end PCDs (use ```log_scaling2.py```)

1. The process is similar to the single log-end PCD scaling, but with extra steps to manage the alignment and positioning of the two PCDs in a global coordinate system, as well as data analysis for the final log scaling report.
2. The global coordinate system is defined as:

    * z-axis is along a vector that is from a leveled ground pointing up toward the sky
    * y-axis is along a vector that is from truck cab to truck bed, e.g. the front-to-end truck axis
    * x-axis is the cross product of the z- and y- axis as defined above
    * the origin (0, 0, 0) is put at the sensor plane intercpt line to the ground plane, at a position along the intercpt line segment

2. Both the frontal and back log-end PCDs are placed in the global coordinate system with its sky-pointing vector along the z-axis (```vec2sky```), looking along its own ```vec2sensor``` direction toward the sensor plane.

## Using The Code

* As a group of Python code scripts for the quick prototype purpose, it is recommended to use the code inside a virtual environment. The code scripts were originally developed under Windows11 + Anaconda virtual environment. The file pathes in the scripts use Windows convention. Change to your case if using Linux.

* Read the code to understand the workflow and necessary parameter setting. The code scripts do not guarrantee the completeness or successful results by any means. Use it at your discretion.

* Any feedback or suggestions would be appreciated!

* Preferred Python version 3.10+ due to the use of ```match ... case ...``` statment. If Python version is lower than 3.10, the ```match ... case ...``` usage in some scripts will need to be modified for compatibility.

* Install necessary packages by using ```pip install -r requirements.txt```

* Use ```python log_loading.py``` to load a mesh or pcd file that is generated by photogrammetry or lidar scan. Follow the terminal prompts to load the raw data, visualize it, and crop the VOI for saving it to a folder at your designation. The croped PCD data is saved as a .PLY PCD file as input dataset for further processing in other scripts.

* Use ```python quick_scaling.py``` to do log scaling on the pcds obtained from the Snoqualmie onsite test datasets, with log diameters and depths info visualized as final results.

* Use ```python log_scaling.py``` to process the pcd within the VOI using Open3D's ```detect_planar_patches()``` method. The current processed results show log end PCD clusters superimposed by rectangle patches with their sizes/depths estimated, surface normal vectors shown, and center postions displayed.

* Use ```python log_scaling2.py``` to process the tow pcd datasets from both log-ends, mimcing the log scaling process [WIP].

* Use ```python log_testing.py``` to try out different ideas in a quick-n-dirty way.

* Use ```python pcd_seg.py``` to play with a variety of segmentation schemes using Open3D or functions from other 3rd-party libraries. Currently, these methods are provided for trials. Be cautious -- This script may have problems with the latest ```pointcloudhandler.py``` and ```utility.py``` due to the contious improvements of the repo.

  * KMEANS from sklearn
  * DBSCAN from sklearn
  * HDBSCAN from hdbscan
  * ECULIDEAN_DBSCAN from Open3D pipeline
  * PLANE_RANSAC_DBSCAN from Open3D pipeline
  * PLANE_RANSAC from Open3D pipeline
  * PLANE_PATCH_PHTGM from Open3D pipeline, with parameter setting optimized for PCD obtained using photogrammetry approach
  * PLANE_PATCH_LIDAR from Open3D pipeline, with parameter setting optimized for PCD acquired using lidar

* Refer to ```pointcloundhandler.py``` and ```utility.py``` for the appropriate class and function, constant definitions.

* Refer to ```poux_pcd_meshing.ipynb``` and ```poux_pcd_segmentation.ipynb``` as two good examples of using Open3D for mesh generation and planar RANSAC+DBSCAN cluster segmenation on some example PCD dataset.

* The ```./data``` folder contains several PCD datasets for running the code scripts. Check each code for details of using these datasets.

## Known Bugs

-[ ] When using ```log_scaling2.py```, the back log-end might not rotate in a right orientation. Will need to debug this issue later.

-[ ] Current ground plane and reference plane detection may give wrong results in some edge cases. Improvements would be needed on these detections when more PCD datasets become available.

## To-do List

-[ ] Fiducial size check for getting physical scaling factor to faciliate the process automation -- write a member func in the PointcloundHandler class

-[ ] Detect/construct ground, reference, sensor planes -- improve the edge case handling

-[ ] Elliptic fitting to get accurate diameters for log-ends -- this seems not the top priority based on the current patch size good results

-[ ] Bark layer thickness segmentation based on pcd color info -- this is a much needed feature to ensure accuracy

-[ ] Put pcds from both ends in the same coordinates based on known reference plane/fiducial orientation and dimensions -- 1st round done, will improve alignment and positioning accuracy, this is a pre-step before the log-end matching

-[ ] Log-end pair matching -- tough nut to crack

-[ ] Further tweak for generalizing the segmenatation and detection parameters -- hold it off until having more datasets to generalize the parameters

-[ ] More PCD visualization features to help with the data analysis and result interpretation -- go along with the code iterations

-[ ] Maybe a simple Open3D GUI to facilitate the workflow -- not needed until next phase

## Note

<steve.yin@freshconsulting.com>, 06/07/2023
