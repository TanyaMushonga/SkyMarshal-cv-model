# SkyMarshal: Intelligent Aerial Traffic Observation System

SkyMarshal is a computer-vision-based traffic monitoring system designed to detect vehicles, track them across frames, and provide real-time speed estimation and Automatic License Plate Recognition (ALPR).

## 🚀 Features

- **Vehicle Detection & Tracking**: Uses YOLOv8 and BoT-SORT to identify and track cars, trucks, motorcycles, and buses with unique IDs.
- **ALPR (Automatic License Plate Recognition)**: Integrated License Plate detection and OCR (EasyOCR) to identify vehicle plates.
- **Speed Estimation**: Uses perspective transformation to map image coordinates to real-world distances for accurate speed calculation.
- **Video Processing**: Batch process video files with detailed visual overlays.

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/TanyaMushonga/SkyMarshal.git
cd SkyMarshal
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚦 Usage

### 1. Prepare Models

- The system uses `yolov8n.pt` for vehicle detection (downloaded automatically).
- **ALPR**: Place a specialized YOLOv8 license plate model named `best.pt` in the project root.

### 2. Run the System

Place your video file in the project directory (e.g., `traffic_sample.mp4`) and run:

```bash
python main.py
```

Results will be saved in the `output/` directory.

## 📂 Project Structure

- `main.py`: Main entry point.
- `src/`:
  - `detector.py`: Vehicle detection logic.
  - `processor.py`: Video stream handling and visualization.
  - `speed_estimator.py`: Perspective-based speed math.
  - `alpr.py`: License plate recognition module.
- `output/`: Processed video results.

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

## 🤝 Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.
