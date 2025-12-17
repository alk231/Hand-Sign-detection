# Hand Sign Detection

Real-time hand gesture recognition using your webcam. Built with Python, OpenCV, and MediaPipe.

## What it does

This project detects hand signs from your camera feed and recognizes different gestures in real-time. It uses MediaPipe's hand tracking to identify hand landmarks and classifies gestures based on finger positions.

Perfect for building sign language translators, gesture-based controls, or just experimenting with computer vision.

## Requirements

```bash
pip install opencv-python mediapipe numpy
```

You'll also need a working webcam.

## Running it

```bash
python main.py
```

Your webcam will open and start detecting hand signs. Press 'q' to quit.

## How it works

The detection pipeline:

1. **Capture video** from your webcam using OpenCV
2. **Detect hands** using MediaPipe's hand detection model
3. **Extract landmarks** - 21 key points on each hand
4. **Classify gesture** based on finger positions and hand orientation
5. **Display result** overlaid on the video feed

## Supported gestures

Depends on what you've trained it on, but common ones include:
- Thumbs up/down
- Open palm
- Fist
- Peace sign
- Pointing
- Number signs (1-5)

You can add your own gestures by collecting training data and updating the classifier.

## Project structure

```
Hand-Sign-detection/
├── main.py              # Main detection script
├── model/               # Trained model files
├── data/                # Training data (if any)
└── utils/               # Helper functions
```

## Training your own gestures

To add custom gestures:

1. Collect images of your hand making the gesture
2. Label the images with the gesture name
3. Extract hand landmarks from each image
4. Train a classifier (SVM, Random Forest, or neural network)
5. Save the model and update the detection script

The more training samples you have, the better the accuracy.

## Tips for better detection

- Use good lighting
- Keep your hand in frame
- Avoid cluttered backgrounds
- Make clear, distinct gestures
- Stay within 1-2 meters from the camera

## Common issues

**Camera not opening**
Check if another app is using your webcam, or try changing the camera index in the code from 0 to 1.

**Low FPS**
MediaPipe can be heavy on some systems. Try reducing the video resolution or running on a machine with better specs.

**Poor detection accuracy**
Make sure you have good lighting and your hand is clearly visible. You might need to retrain the model with more diverse data.

## Tech stack

- **OpenCV** - Video capture and image processing
- **MediaPipe** - Hand tracking and landmark detection
- **NumPy** - Data manipulation
- **scikit-learn** or **TensorFlow** - Gesture classification (depending on implementation)

## Possible improvements

- Add more gesture classes
- Improve classifier accuracy
- Add gesture history tracking
- Build a GUI with gesture stats
- Create a dataset collection tool
- Add audio feedback for detected gestures
- Support two-hand gestures

## Credits

Built using Google's MediaPipe framework for hand tracking.