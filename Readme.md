# 🎾 AI Tennis Analysis System

An **AI-powered Computer Vision project** that brings professional-level **tennis match analysis** to your screen. This system combines **object detection**, **pose estimation**, **3D visualization**, and **physics-based ball tracking** to analyze and visualize tennis gameplay in real-time.

---

## 📋 Features

- **🎯 Real-Time Player Detection & Tracking**  
  Detects and tracks players using YOLOv11 and a lightweight tracking pipeline.

- **🏸 Tennis Ball Tracking & Trajectory Prediction**  
  Detects and predicts the tennis ball's motion, including bounces and height variation.

- **🧍‍♂️ Player Pose Estimation (Skeleton Drawing)**  
  Draws 2D and 3D skeletons for players only, filtering out non-player detections (e.g., officials).

- **🌍 3D Court Visualization**  
  Renders a real-world tennis court in 3D with ball physics simulation and synchronized player motion.

- **📊 Speed & Match Statistics Overlay**  
  Displays player and ball speeds (in km/h) using calibrated scaling from court geometry.

- **🎥 Side-by-Side Visualization**  
  Combines real video frames with the 3D rendered match scene for a complete analytical view.

---

## ⚙️ Tech Stack

| Category | Technologies |
|-----------|--------------|
| **AI/ML Models** | YOLOv11, DeepSORT (custom tracking) |
| **Computer Vision** | OpenCV, Matplotlib 3D |
| **Programming** | Python, NumPy, PyTorch |
| **Visualization** | Matplotlib, CV2 HUD overlays |
| **Data Sources** | Roboflow, Kaggle |

---

## 📦 Datasets

- **🎾 Tennis Ball Detection Dataset:** Combined datasets from [Roboflow](https://roboflow.com) and Kaggle for robust training.
- **🏟 Court Detection Dataset:** [Tennis Court Detection Dataset on Roboflow](https://universe.roboflow.com/hautrinhvan/tenniscourtdetection/dataset/3)

All models and weights are provided within the GitHub repository.

---

## 🧠 Project Structure

```
AI-Tennis-Analysis/
├── src/
│   ├── ball_tracker.py
│   ├── people_overlay.py
│   ├── court_pose.py
│   ├── plot3d.py
│   ├── draw.py
│   ├── lifter3d.py
│   ├── auto_classmap.py
│   ├── video_io.py
│   └── __init__.py
├── config/
│   ├── settings.yaml
│   └── settings_example.yaml
├── run_ball_people& court_pose.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/ai-tennis-analysis.git
cd ai-tennis-analysis
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Paths
Edit your `config/settings.yaml` file with paths for your video input, models, and output directory.

### 5️⃣ Run the Project
```bash
python run_ball_people& court_pose.py
```

### 6️⃣ Output
A combined side-by-side video will be saved in your output path with real-time player, ball, and 3D analysis.

---

## 📊 Example Output

- **Left Side:** Original video with player tracking, ball trajectory, and info overlay.  
- **Right Side:** Real-time 3D tennis court visualization.

---

## 🧾 License
This project is open-source under the **MIT License**.

---

## 🙌 Acknowledgements

Special thanks to:
- [Ultralytics](https://github.com/ultralytics) for YOLOv11
- [Roboflow](https://roboflow.com) for datasets and tools
- [Kaggle](https://www.kaggle.com) for data resources
- [OpenCV](https://opencv.org) for real-time computer vision utilities

---

## 🏷 Hashtags
#ArtificialIntelligence #ComputerVision #SportsAnalytics #Tennis #MachineLearning #YOLO #DeepLearning #OpenCV #RealTimeAnalytics #SportsTech #TechInnovation #DataScience #VideoAnalysis #FutureOfSports #DigitalTransformation #Python #AI #SportsEngineering #SmartAnalytics #3DTennis #SportScience #PerformanceTracking #SportTechnology

