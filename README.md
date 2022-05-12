# Uno Cards

Recognizing uno cards using computer vision
- Project source: https://labcode.mdx.ac.uk/alinivan/uno-cards
- Real Images datasets: https://labcode.mdx.ac.uk/alinivan/uno-cards/-/tree/main/img
- Presentation video: TBA

# Getting started
**Additional documentation available in the documentation folder**

### Pre-requisites
 - Python 3.9.7
 - Numpy 1.20.3
 - OpenCV 4.5.5
 - TensorFlow 2.8
 - Matplotlib 


### Run on a local machine

In a terminal window
```
git clone https://labcode.mdx.ac.uk/alinivan/uno-cards.git
cd uno-cards
python main.py
```
When running a menu will appear in the terminal window as ilustrated bellow.
```
-----------------
*** Uno Cards ***
-----------------
[1] Read from file
[2] Live camera stream
[3] Data capturing
[0] Exit
```

# Program logic

**Option 1 Read from file**
 - Opens submenu with 2 options: read data from file **by selection or** read **all data** from files
 - Performs image processing and outputs image data (color and number)
 - **Reliant on .jpg image data in img** folder


**Option 2 - Live camera stream**
 - Starts a camera stream
 - Performs image processing and outputs image data (color and number)
 - **Reliant on a camera** connected to the machine

**Option 3 - Data capturing** **DEVELOPMENT ONLY**
 - Runs a script that captures uno cards images
 - May overwrite the images files in the img folder within th eproject source

# Files and Folders description and requirements

**Source folder**
 - **sensible structure of folders and python files**

**Folders:**
- **img** folder contains the **uno cards images**

**Python files:**
- **main.py** handles the **program logic** according to the selected menu options
- **core.py** handles the abstract **image recognition**
- **image_processing.py** handles **image processing** and recognition of **color**, **number** of the each card
- **data_capturing.py** handles the **capturing of the images for static processing**

## Project info
- University project
- MSc Robotics
- Computer Vision and Image Recognition

## Authors and acknowledgment
- Alin Ivan
- Middlesex University
- 2022

## License
- MIT
