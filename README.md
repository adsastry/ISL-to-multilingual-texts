# ISL-to-multilingual-texts
Helpful to the dumb and deaf community of India who uses Indian Sign Language to communicate.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2
image_size = 128
batch_size = 32
channels = 3
epochs = 20
video_dir = "path_to_your_dataset"

image_size = 128  
batch_size = 32
num_frames = 30 

def load_video(video_path, num_frames, image_size):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, num_frames).astype(int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (image_size, image_size))
        frame = frame / 255.0  # Normalize
        frames.append(frame)

    cap.release()
    
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))  # Pad if needed
    
    return np.array(frames)

def load_video_dataset(video_dir, num_frames, image_size):
    videos = []
    labels = []
    class_names = sorted(os.listdir(video_dir))  # Get class labels
    
    for class_index, class_name in enumerate(class_names):
        class_path = os.path.join(video_dir, class_name)
        for video_file in os.listdir(class_path):
            video_path = os.path.join(class_path, video_file)
            video_data = load_video(video_path, num_frames, image_size)
            videos.append(video_data)
            labels.append(class_index)

    return np.array(videos), np.array(labels), class_names

X, y, class_names = load_video_dataset(video_dir, num_frames, image_size)

dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(len(X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print(f"Dataset size: {len(X)} videos")
print(f"Number of classes: {len(class_names)}")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization

num_classes = len(class_names)  
input_shape = (num_frames, image_size, image_size, 3)  

model = Sequential([
    Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
    MaxPooling3D((2, 2, 2)),

    Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
    MaxPooling3D((2, 2, 2)),

    Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
    MaxPooling3D((2, 2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
