# 🏃‍♂️ **Player Re-Identification in Sports Footage**

This project implements a robust **computer vision system** for tracking and ***re-identifying players*** in sports videos. The system maintains consistent player IDs even when players **exit and re-enter** the frame—ensuring smooth and intelligent tracking throughout the video.

> 📌 Built for sports analytics, player monitoring, and intelligent broadcasting use-cases.


## 🚀 **Key Features**

- 🎯 **Real-Time Player Detection** using YOLOv11  
- 🔁 **Multi-Object Tracking** across consecutive video frames  
- 🧠 **Re-identification Module** handles occlusions and re-entries  
- 🎥 **Video Output** with annotated bounding boxes and player IDs  
- ⚡ Optional support for **CUDA GPU acceleration**


## 📁 **Project Structure**
player-reidentification/
├── .vscode/ # Optional: VS Code workspace settings
│ └── settings.json
│
├── data/
│ └── 15sec_input_720p.mp4 # Input video
│
├── models/
│ └── yolov11_model.pt # YOLOv11 pretrained model
│
├── main.py # Main script for detection & tracking
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Ignored files & folders

> 🗂️ Folders like `output/`, `__pycache__/`, and `.env/` are auto-generated and excluded via `.gitignore`.



## ⚙️ **Installation & Setup**
### 📌 **Prerequisites**

- **Python 3.8+**
- **pip**
- **Git** (optional)
- **CUDA-compatible GPU** (optional, but recommended)


### 🧪 **Installation Steps**

bash
## Clone the repository
git clone https://github.com/Rahuljoshi1216/reidentification-player.git
cd reidentification-player

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


### 🎬 Section 5: **How to Run**
## 🎬 **How to Run**

1. Place your input `.mp4` video inside the `data/` folder  
2. Place your YOLOv11 `.pt` model file inside the `models/` folder  
3. Then run:

`bash
python main.py


## ⚙️ **Tracker Customization**

You can tweak the tracking and re-identification logic directly in `main.py`. For example:

`python`
tracker = PlayerTracker(
    max_disappeared_frames=30,
    distance_threshold=100,
    reidentification_threshold=80
)


`markdown`
## 📦 **Dependencies**

Core libraries used in this project:

- `ultralytics` — YOLOv11 detection  
- `opencv-python` — video frame handling and drawing  
- `numpy` — numerical operations  
- `torch`, `torchvision` — PyTorch backend

All are listed in `requirements.txt`.


## 🧪 **Troubleshooting & Tips**

| **Issue**                      | **Solution**                                                           |
|-------------------------------|------------------------------------------------------------------------|
| File not found (video/model)  | Double-check file names & folder paths                                 |
| YOLO model too large          | Consider using a smaller model or running with GPU                     |
| Tracking inaccurate           | Tweak `distance_threshold` or confidence filters in `main.py`          |
| Slow performance              | Enable GPU (CUDA) and use a smaller video resolution                   |


## 🌐 **Handling Large Files (YOLO model & Videos)**

Large files like `.pt` models and `.mp4` videos are not stored directly in this repository.

Instead, you can download them from the following Google Drive links:


### 📥 **Downloads**

- 🎥 **Test Video Folder** (contains sample `.mp4`):
  [Google Drive - Videos](https://drive.google.com/drive/folders/1Nx6H_n0UUI6L-6i8WknXd4Cv2c3VjZTP)

- 🧠 **YOLOv11 Pretrained Model** (`yolov11_model.pt`):
  [Google Drive - YOLO Model](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)

### 📌 **Usage Instructions**

1. **Download** the video from the folder and place it in: data/
2. **Download** the YOLO model and place it in: models/


## 📜 **License**

This project is intended for **educational and research purposes only**.  
Ensure you have the legal rights to use and share any video inputs.


## 📬 **Contact & Contributions**

Got suggestions, bugs, or want to contribute?  
Feel free to [open an issue](https://github.com/Rahuljoshi1216/reidentification-player/issues) or create a pull request.
