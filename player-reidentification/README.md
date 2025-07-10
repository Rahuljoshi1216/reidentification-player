# Player Re-Identification in Sports Footage 🎯

This project implements a computer vision system for tracking and re-identifying players in sports footage. The system can maintain consistent player IDs even when players temporarily leave the frame and return.

---

## 🚀 Features

- 🎯 **Player Detection** using YOLOv11
- 🔁 **Player Tracking** across consecutive frames
- 🧠 **Re-identification** after occlusions or re-entries
- 🎥 **Video Output** with annotated bounding boxes and player IDs

---

## 📁 Project Structure

```
player-reidentification/
├── .vscode/
│   └── settings.json              # VS Code workspace settings (optional)
│
├── data/
│   └── 15sec_input_720p.mp4       # Input video for testing
│
├── models/
│   └── yolov11_model.pt           # Pre-trained YOLOv11 model
│
├── main.py                        # Main script to run detection and tracking
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignored files and folders
```

> 🔥 Note: `output/` and `__pycache__/` directories are ignored via `.gitignore` and will be auto-generated during execution.

---

## ⚙️ Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/Rahuljoshi1216/player-reidentification.git
cd player-reidentification

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎬 Usage

Place your input video and YOLOv11 model in the respective folders (`data/`, `models/`). Then run:

```bash
python main.py
```

This will process the video and output an annotated version with consistent player tracking.

---

## 🛠️ Customization

Modify the tracking logic or parameters inside `main.py` as needed:

```python
tracker = PlayerTracker(
    max_disappeared_frames=30,
    distance_threshold=100,
    reidentification_threshold=80
)
```

---

## 📦 Dependencies

- `ultralytics`
- `opencv-python`
- `numpy`
- `torch`
- `torchvision`

---

## 🧪 Troubleshooting

### Common Issues

- **Video/Model Not Found**: Ensure files are in correct folders and names match
- **Low Accuracy**: Tweak confidence and distance thresholds in tracker settings

---

## 📌 Notes

- Make sure to enable GPU (CUDA) if available for better performance
- For large models or videos, monitor memory usage
- Avoid committing large `.pt` or `.mp4` files unless necessary

---

## 📜 License

This project is for educational and research purposes only. Ensure you have rights to any input video used.

---

## 📬 Contact

For issues, suggestions, or contributions, feel free to open an issue or PR.
