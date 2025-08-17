# Feature Matching and Epipolar Geometry Estimation

This project implements feature matching between two images, computes the fundamental matrix, and visualizes epipolar lines using different feature detectors and descriptor matchers. It also recovers the relative camera pose and calculates Euler angles.

## Features

- Multiple feature detectors supported: **ORB, AKAZE, BRISK, KAZE, SIFT, FAST+BRIEF**
- FLANN-based and LSH-based descriptor matching
- RANSAC-based fundamental matrix estimation
- Essential matrix computation and pose recovery
- Epipolar line visualization
- Euler angles calculation (Roll, Pitch, Yaw)

## Project Structure

```
├── CMakeLists.txt     # Build configuration using CMake
├── config.json        # Camera calibration matrices and LSH parameters
├── main.cpp           # Main program
├── external/         
│   ├── json/
│       ├── json.hpp      # header file of json library
├── imgs_kiti/         # (You need to place your image files here)
│   ├── scene-1/
│       ├── F.jpg      # First view image
│       ├── L.jpg      # Second view image
```

## Getting Started

### Prerequisites

- CMake >= 3.10
- C++17 Compiler
- OpenCV (with `xfeatures2d` and `flann` modules)
- `nlohmann/json` library (for reading `config.json`)

### Building

```bash
mkdir build
cd build
cmake ..
make
```

### Running

Make sure the `config.json` and image files are correctly placed relative to the executable. Then run:

```bash
./FeatureMatchingEpipolar
```

## Configuration

The `config.json` file contains:

- `camera_matrices`: Intrinsic matrices (`K1`, `K2`) for both cameras.
- `lsh_params`: Parameters for LSH-based matching (`table_number`, `key_size`, `multi_probe_level`).

Example:

```json
{
  "camera_matrices": {
    "K1": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "K2": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  },
  "lsh_params": {
    "table_number": 20,
    "key_size": 24,
    "multi_probe_level": 4
  }
}
```

## Notes

- Make sure SIFT and other patented algorithms are enabled in your OpenCV build (via `opencv_contrib`).
- Images should be grayscale (`IMREAD_GRAYSCALE` is used).
- If the number of good matches is too low, pose recovery will be skipped.

