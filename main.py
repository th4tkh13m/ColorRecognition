import cv2
import json
import numpy as np
from sklearn.cluster import KMeans
import webcolors

with open("newcolors.json") as f:
    colors = json.load(f)
    


# Convert hex colors to RGB values for cluster center initialization
INITIAL_CLUSTER_CENTERS = [webcolors.hex_to_rgb(hex_value) for hex_value in colors.keys()]

# Initialize K-Means with predefined RGB values as initial cluster centers
kmeans = KMeans(n_clusters=len(colors), n_init=1)
kmeans.fit(INITIAL_CLUSTER_CENTERS)  # Fit the model to the data

def get_color_name(rgb_tuple):
    return colors.get(webcolors.rgb_to_hex(rgb_tuple).upper(), "Unknown")

def extract_main_colors(model, frame, num_colors=3):
    # Resize the frame for faster processing if needed
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Reshape the frame to a list of pixels
    pixels = frame.reshape((-1, 3))
    
    
    # Predict the labels for each pixel based on the fitted model
    labels = model.predict(pixels)
    
    # Count the number of pixels assigned to each cluster
    cluster_counts = np.bincount(labels, minlength=num_colors)
    
    # Get the indices of the top cluster centers with most points
    top_cluster_indices = cluster_counts.argsort()[-num_colors:][::-1]
    
    # Get the RGB values of the top cluster centers
    top_cluster_centers_rgb = kmeans.cluster_centers_[top_cluster_indices].astype(int)

    # Convert RGB values to color names
    top_cluster_colors_names = [get_color_name(tuple(rgb)) for rgb in top_cluster_centers_rgb]
    
    return top_cluster_colors_names, top_cluster_centers_rgb


# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    top_cluster_colors_names, top_cluster_centers_rgb = extract_main_colors(kmeans, frame)
    
    for i, (name, rgb) in enumerate(zip(top_cluster_colors_names, top_cluster_centers_rgb)):
        text = f"{name} ({rgb})"
        bgr_color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))  # Convert RGB to BGR
        position = (10, 30 + i * 30)  # Adjust vertical position for each color
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, bgr_color, 2)
    
    
    # Display the frame
    cv2.imshow("Webcam", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
