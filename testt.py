import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine, euclidean, cityblock
import os

class HistogramLogoMatcher:
    """
    A class for comparing logo similarity using color and edge histograms.
    This approach complements deep learning methods by using traditional
    computer vision techniques that can be more robust to certain transformations.
    """
    
    def __init__(self, bins=32, use_hsv=True, edge_bins=16):
        """
        Initialize the logo matcher with histogram parameters.
        
        Args:
            bins: Number of bins for color histogram
            use_hsv: Whether to use HSV color space (better for color similarity)
            edge_bins: Number of bins for edge orientation histogram
        """
        self.bins = bins
        self.use_hsv = use_hsv
        self.edge_bins = edge_bins
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess an image for histogram extraction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (color_img, gray_img) for histogram computation
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Could not read image at {image_path}")
            
        # Resize for consistency
        img = cv2.resize(img, (224, 224))
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Convert to HSV if needed
        if self.use_hsv:
            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            color_img = img  # Use BGR
            
        return color_img, gray
    
    def extract_color_histogram(self, color_img):
        """
        Extract color histogram features from an image.
        
        Args:
            color_img: Preprocessed color image (HSV or BGR)
            
        Returns:
            Normalized color histogram
        """
        # Create range for each channel
        if self.use_hsv:
            ranges = [180, 256, 256]  # H: 0-179, S: 0-255, V: 0-255
        else:
            ranges = [256, 256, 256]  # B, G, R: 0-255
            
        # Create histogram
        hist = cv2.calcHist(
            [color_img], 
            [0, 1, 2],  # All channels
            None,  # No mask
            [self.bins, self.bins, self.bins],  # Bins per channel
            [0, ranges[0], 0, ranges[1], 0, ranges[2]]  # Ranges
        )
        
        # Normalize histogram
        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        return hist.flatten()
    
    def extract_edge_histogram(self, gray_img):
        """
        Extract edge orientation histogram features.
        
        Args:
            gray_img: Grayscale image
            
        Returns:
            Normalized edge orientation histogram
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        
        # Calculate gradients using Sobel
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and orientation
        magnitude = cv2.magnitude(grad_x, grad_y)
        orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        
        # Create a mask for significant edges
        # This ignores weak edges that might be noise
        mask = magnitude > 50
        
        # Extract histogram of orientations for significant edges
        edge_hist, _ = np.histogram(
            orientation[mask], 
            bins=self.edge_bins,
            range=(0, 180)
        )
        
        # Normalize
        if np.sum(edge_hist) > 0:  # Avoid division by zero
            edge_hist = edge_hist / np.sum(edge_hist)
            
        return edge_hist
    
    def extract_shape_features(self, gray_img):
        """
        Extract shape-based features using contours.
        
        Args:
            gray_img: Grayscale image
            
        Returns:
            Shape descriptor vector
        """
        # Apply thresholding
        _, thresh = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, return zeros
        if not contours:
            return np.zeros(7)
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Extract shape features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Avoid division by zero
        if perimeter == 0:
            circularity = 0
        else:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        extent = area / (w * h) if w * h > 0 else 0
        
        # Convex hull
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Moments for center of mass
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center_dist = np.sqrt((cx - 112)**2 + (cy - 112)**2) / 112  # Normalized distance from center
        else:
            center_dist = 1
            
        # Return shape features
        return np.array([
            circularity, 
            aspect_ratio, 
            extent, 
            solidity, 
            perimeter / 100,  # Scaled perimeter
            area / 10000,     # Scaled area
            center_dist       # Distance from center
        ])
    
    def extract_features(self, image_name):
        """
        Extract all features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary of features
        """
        color_img, gray_img = self.preprocess_image(f'images/{image_name}')
        
        blurred_color = cv2.GaussianBlur(color_img, (9, 9), 0)
        blurred_gray = cv2.GaussianBlur(gray_img, (9, 9), 0)

        color_hist = self.extract_color_histogram(blurred_color)
        edge_hist = self.extract_edge_histogram(blurred_gray)
        shape_features = self.extract_shape_features(blurred_gray)
        
        return {
            'color_hist': color_hist,
            'edge_hist': edge_hist,
            'shape_features': shape_features
        }
    
    def compute_similarity(self, features1, features2, weights=None):
        """
        Compute similarity between two feature sets.
        
        Args:
            features1: First feature set
            features2: Second feature set
            weights: Dictionary with weights for each feature type
                    (default: color=0.4, edge=0.4, shape=0.2)
                    
        Returns:
            Similarity score (higher means more similar)
        """
        if weights is None:
            weights = {'color': 0.1, 'edge': 0.1, 'shape': 0.8}
        
        # Compute histogram correlation for color (higher is better)
        color_sim = 1 - cosine(features1['color_hist'], features2['color_hist'])
        
        # Compute histogram correlation for edges (higher is better)
        edge_sim = 1 - cosine(features1['edge_hist'], features2['edge_hist'])
        
        # Compute Euclidean distance for shape features (lower is better)
        shape_dist = euclidean(features1['shape_features'], features2['shape_features'])
        # Convert to similarity (higher is better)
        shape_sim = 1 / (1 + shape_dist)
        
        # Weighted sum
        similarity = (
            weights['color'] * color_sim + 
            weights['edge'] * edge_sim + 
            weights['shape'] * shape_sim
        )
        
        return similarity


# Example usage
if __name__ == "__main__":
    # Initialize matcher
    matcher = HistogramLogoMatcher(bins=16, use_hsv=True, edge_bins=18)
    
    # Compare two logos
    logo1_path = "1414.png"
    logo2_path = "158.png"
    
    # # Only run this with valid file paths
    # similarity, details = matcher.compare_logos(
    #     logo1_path, 
    #     logo2_path,
    #     weights={'color': 0.4, 'edge': 0.4, 'shape': 0.2},
    #     show_visualization=True
    # )
    
    # print(f"Overall similarity: {similarity:.3f}")
    # print(f"Color similarity: {details['color_similarity']:.3f}")
    # print(f"Edge similarity: {details['edge_similarity']:.3f}")
    # print(f"Shape similarity: {details['shape_similarity']:.3f}")
    
    # Find similar logos in a directory
    # similar_logos = matcher.find_similar_logos(logo1_path, "path/to/logo/directory", top_k=5)
    # print("\nSimilar logos:")
    # for logo_path, sim in similar_logos:
    #     print(f"{os.path.basename(logo_path)}: {sim:.3f}")