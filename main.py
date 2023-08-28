import cv2
from recognizer import ColorRecognizer

color_recognizer = ColorRecognizer("newcolors.json")


# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    top_cluster_colors_names, top_cluster_centers_rgb = color_recognizer.extract_main_colors(frame)
    
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
