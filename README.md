# ISL-to-multilingual-texts
Helpful to the dumb and deaf community of India who uses Indian Sign Language to communicate.
This project is a deep learning-based solution for recognizing Indian Sign Language (ISL) dynamic gestures from video input. It uses a 3D Convolutional Neural Network (CNN) to classify gestures and Streamlit to provide a web-based interface. The recognized gesture is also translated into Hindi and Kannada using Google Translate.

Unfortunately, we are unable to upload the dataset due to it's ize. The link to the dataset is provided below. There aren't many datasets which has all the gestures. The dataset which we used are the emergency sings.
https://data.mendeley.com/datasets/2vfdm42337/1

# Model Architecture
Input: Video sequence of shape `(30 frames, 128x128 pixels, 3 channels)`

Network: 3D CNN with Conv3D, MaxPooling3D, Dropout, and Dense layers

Output: Gesture class (e.g., Call, Help, Doctor) with softmax confidence

Languages Supported: English ➡ Hindi, Kannada

# Technologies Used
Python

Convolutional Neural Network

TensorFlow / Keras

OpenCV

Streamlit

Googletrans (Translation)

# Key Features
Real time gesture prediction

AI powered algorithm for intelligent data processing

Friendly user interface

# Preview
1. Main UI
<img width="1239" height="539" alt="image" src="https://github.com/user-attachments/assets/e6cae22b-3165-43a1-a1a4-2773b2428bb7" />

2. Video uploaded 
<img width="946" height="726" alt="image" src="https://github.com/user-attachments/assets/5b32964e-ac73-49af-b2ec-9316261820d9" />

3. Final Output
<img width="918" height="296" alt="image" src="https://github.com/user-attachments/assets/6c6ebbb7-7aae-4ca6-9a0d-b2a9eeed9636" />

The research paper associated with this project can be found by the following link:
https://www.researchgate.net/publication/393437067_Indian_Sign_Language_to_Multilingual_Text_Using_Deep_Learning

# Contributors
Sharon Sara

Shubha M
