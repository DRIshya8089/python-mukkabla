🪞 MUKKABLA EFFECT USING PYTHON & OPEN-CV
📖 Overview

This project creates a real-time "skin invisibility" effect using a webcam.
When a person stands in front of the camera, their skin areas (face, arms, hands, legs) are detected and replaced with the background, making them appear invisible — while clothes, hair, and other non-skin parts remain visible.

It’s like a custom “invisible cloak,” but applied only to human skin portions.

⚙️ Features

Real-time webcam feed processing

Background capture (press b to save a clean plate)

Human parsing segmentation (to isolate skin regions)

Skin area masking & replacement with background

Edge smoothing and temporal filtering for natural results

Quit anytime with q

🛠️ Tools & Libraries Used

Python 3.10+

OpenCV → Webcam capture, image processing, compositing

OpenCV-contrib → Extra computer vision modules

NumPy → Array & numerical operations

ONNX / ONNX Runtime → For running the human-parsing segmentation model

Pillow → For image handling

Matplotlib → For debugging/visualization

tqdm → For progress indication (when loading data or testing)
