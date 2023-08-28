import cv2
import json
import numpy as np
from sklearn.cluster import KMeans
import webcolors

class ColorRecognizer:
    def __init__(self, colors_file: str) -> None:
        """
        Initialize the ColorRecognizer class.

        Parameters:
            colors_file (str): Path to the JSON file containing color information.
        """
        with open(colors_file) as f:
            self.colors = json.load(f)
        self.model = self.define_model(self.colors)

    def get_color_name(self, rgb_tuple: (int, int, int)) -> str:
        """
        Convert an RGB tuple to a color name.

        Parameters:
            rgb_tuple (tuple): RGB values as a tuple (R, G, B).

        Returns:
            str: Color name corresponding to the RGB values.
        """
        return self.colors.get(webcolors.rgb_to_hex(rgb_tuple).upper(), "Unknown")

    @staticmethod
    def define_model(colors: dict[str, str]) -> KMeans:
        """
        Define and initialize the K-Means clustering model.

        Parameters:
            colors (dict): Dictionary containing color hex codes as keys and color names as values.

        Returns:
            KMeans: Initialized K-Means clustering model.
        """
        # Convert hex colors to RGB values for cluster center initialization
        INITIAL_CLUSTER_CENTERS = [webcolors.hex_to_rgb(hex_value) for hex_value in colors.keys()]

        # Initialize K-Means with predefined RGB values as initial cluster centers
        kmeans = KMeans(n_clusters=len(colors), n_init=1)
        kmeans.fit(INITIAL_CLUSTER_CENTERS)  # Fit the model to the data
        
        return kmeans

    def extract_main_colors(self, frame: np.ndarray, size: (int, int) = (640, 480), num_colors: int = 3) -> ([str], np.ndarray):
        """
        Extract the main colors from an image using the K-Means model.

        Parameters:
            frame (numpy.ndarray): Input image as a NumPy array.
            size (tuple): Size to resize the frame for processing (default is (640, 480)).
            num_colors (int): Number of main colors to extract (default is 3).

        Returns:
            list: List of color names.
            numpy.ndarray: Array of RGB values corresponding to the extracted main colors.
        """
        # Resize the input frame for faster processing if needed
        frame = cv2.resize(frame, size)

        # Convert the BGR image to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Reshape the image pixels to a flat array
        pixels = frame.reshape((-1, 3))

        # Predict cluster labels for each pixel using the K-Means model
        labels = self.model.predict(pixels)

        # Count the number of pixels assigned to each cluster
        cluster_counts = np.bincount(labels, minlength=num_colors)

        # Get the indices of the clusters with the most points
        top_cluster_indices = cluster_counts.argsort()[-num_colors:][::-1]

        # Get the RGB values of the top cluster centers
        top_cluster_centers_rgb = self.model.cluster_centers_[top_cluster_indices].astype(int)

        # Convert RGB values to color names using the predefined colors dictionary
        top_cluster_colors_names = [self.get_color_name(tuple(rgb)) for rgb in top_cluster_centers_rgb]

        return top_cluster_colors_names, top_cluster_centers_rgb
