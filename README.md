# ISL-to-multilingual-texts
Helpful to the dumb and deaf community of India who uses Indian Sign Language to communicate.
This project is a deep learning-based solution for recognizing Indian Sign Language (ISL) dynamic gestures from video input. It uses a 3D Convolutional Neural Network (CNN) to classify gestures and Streamlit to provide a web-based interface. The recognized gesture is also translated into Hindi and Kannada using Google Translate.

Model Architecture

Input: Video sequence of shape `(30 frames, 128x128 pixels, 3 channels)`
Network: 3D CNN with Conv3D, MaxPooling3D, Dropout, and Dense layers
Output: Gesture class (e.g., Call, Help, Doctor) with softmax confidence
Languages Supported: English ➡ Hindi, Kannada

Technologies Used

Python
TensorFlow / Keras
OpenCV
Streamlit
Googletrans (Translation)
NumPy
