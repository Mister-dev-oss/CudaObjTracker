# CUDA Object Tracker

Real-time object tracking system using CUDA acceleration, without neural networks.

## 🚧 Project Status

**Work in Progress** - Currently implementing the processing pipeline.

The tracking algorithm and feature detection method are still being evaluated. This project focuses on classical computer vision techniques accelerated with CUDA.

## ✅ Current Features

- [x] GPU-accelerated grayscale conversion using NPP (NVIDIA Performance Primitives)
- [x] Triple buffering pipeline (capture → pinned memory → GPU)
- [ ] Image preprocessing (currently doing, for now only greyscale)
- [ ] Feature detection (ORB/SIFT/Harris - TBD)
- [ ] Feature matching and tracking
- [ ] Object localization and bounding boxes

## 🎯 Project Goals

Build a real-time object tracker using:
- **CUDA** for GPU acceleration
- **Classical CV algorithms** (no neural networks)
- **Efficient memory management** (pinned memory, async transfers)

## 📋 Requirements

### Hardware
- NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- Webcam with **MJPEG support** (required for proper performance)

### Software
- CUDA Toolkit 11.0+ (tested with 12.x)
- OpenCV 4.x with V4L2 support
- NPP (NVIDIA Performance Primitives) - included with CUDA Toolkit
- g++ with C++14 support

### Libraries
- `libopencv-dev` (or `opencv4`)
- `cuda-toolkit`
- NPP libraries: `nppc`, `nppig`, `nppicc`

## 🛠️ Installation

### Ubuntu/Debian
```bash
# Install dependencies
sudo apt update
sudo apt install build-essential libopencv-dev

# CUDA Toolkit (if not already installed)
# Follow: https://developer.nvidia.com/cuda-downloads
```

### Build
```bash
# Clone repository
git clone https://github.com/Mister-dev-oss/CudaObjTracker.git
cd CudaObjTracker

# Compile
make

# Clean (if needed)
make clean
```

## 🚀 Usage

```bash
# Default resolution (webcam default)
# WILL USE RAW DATA, NOT COMPRESSED FORMATS ==> LOWER FPS ON BIG RESOLUTIONS
./greyscale

# Custom resolution (width height)
# WIlL USE MJPG IF SUPPORTED
./greyscale 1920 1080
./greyscale 1280 720
```

### Important Notes
- Your webcam **must support MJPEG** format for optimal performance
- Press `ESC` to exit

## 📊 Performance

Actual framerate depends on:
- GPU model
- Camera resolution
- Memory bandwidth

## 🏗️ Architecture ( !! FOR NOW !! )

```
Capture Thread  →  [Pageable Buffer]  (OpenCV capture)
                          ↓
Pinned Thread   →  [Pinned Buffer]    (Fast CPU memory)
                          ↓
Main Thread     →  [GPU Processing]   (CUDA/NPP kernels)
                          ↓
                    [Display]          (OpenCV imshow)
```

### Memory Flow
1. **Capture**: Webcam → pageable memory (double buffered)
2. **Copy**: Pageable → pinned memory (double buffered)
3. **Transfer**: Pinned CPU → GPU (async via CUDA streams) (async currently does not provide any performance increment)
4. **Process**: Grayscale conversion on GPU (NPP)
5. **Retrieve**: GPU → CPU (sync copy)
6. **Display**: OpenCV window

## 📁 Project Structure

```
cuda-tracker/
├── Makefile
├── greyscale.cu         # Main processing pipeline
├── README.md
└── .gitignore
```

## 🔧 Configuration

### Camera Settings
The program attempts to set MJPEG format automatically:
```cpp
cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
```

If your camera doesn't support MJPEG, the program will fall back to the default format (may reduce performance based on resolution and hardware (camera,drivers...)).

### Check Camera Capabilities
```bash
# List available cameras
v4l2-ctl --list-devices

# Check supported formats for /dev/video0
v4l2-ctl -d /dev/video0 --list-formats-ext
```

## 🐛 Troubleshooting

### "Webcam cannot be opened"
- Check camera permissions: `sudo chmod 666 /dev/video0`
- Verify camera is detected: `ls /dev/video*`

### Low FPS / Performance Issues
- Ensure MJPEG is supported by your camera
- Lower resolution: `./greyscale 640 480`
- Check CUDA installation: `nvidia-smi`

### Compilation Errors
- Verify CUDA installation: `nvcc --version`
- Check OpenCV: `pkg-config --modversion opencv4`
- Ensure NPP libraries are installed (part of CUDA Toolkit)
- Check CUDA path in Makefile (default: `/usr/local/cuda`)

## 📚 Next Steps

Pipeline decisions to be made:
- **Feature detector**: ORB, SIFT, FAST, Harris corners?
- **Tracking method**: Optical flow, feature matching, Kalman filtering?
- **Multi-object**: Single vs multiple object tracking?

All implementations will use CUDA acceleration where possible.

## 📄 License

MIT License

## 🤝 Contributing

This is a learning/experimental project. Suggestions and contributions are welcome!

**Note**: This project is under active development. Architecture may change.
