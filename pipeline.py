import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup, Comment
import re
import html
import cairosvg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import json
from tqdm import tqdm
import google.generativeai as genai 
from dotenv import load_dotenv
import shutil as shutuil
import cv2
import numpy as np
from sklearn.neighbors import BallTree
from collections import Counter
import sys

class LogoNet(nn.Module):
    """
    A specialized neural network for logo recognition with:
    1. Enhanced local feature extraction
    2. Multi-scale processing
    3. Shape-aware attention
    4. Rotation and scale invariance
    """
    def __init__(self, embedding_dim=512):
        """
        Initialize the LogoNet model with specialized architecture for logo recognition.
        
        Args:
            embedding_dim (int): Dimension of the output embedding vector. Default is 512.
        """
        super(LogoNet, self).__init__()
        
        # Base feature extractor (EfficientNet is good for logos due to better edge detection)
        # Could use ResNet50 or other backbones too
        self.backbone = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)
        backbone_out_features = 2048  # EfficientNet-B5 output features
        
        # Remove classifier head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Multi-scale feature extraction (helps with logos of different scales)
        self.conv_1x1 = nn.Conv2d(backbone_out_features, 256, kernel_size=1)
        self.conv_3x3 = nn.Conv2d(backbone_out_features, 256, kernel_size=3, padding=1)
        self.conv_5x5 = nn.Conv2d(backbone_out_features, 256, kernel_size=5, padding=2)
        
        # Shape-aware attention module
        self.attention = nn.Sequential(
            nn.Conv2d(768, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 768, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global context module
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(768, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Forward pass of the LogoNet model.
        
        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Logo embedding vector
        """
        # Extract base features
        x = self.backbone(x)
        
        # Multi-scale feature extraction
        feat_1x1 = self.conv_1x1(x)
        feat_3x3 = self.conv_3x3(x)
        feat_5x5 = self.conv_5x5(x)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat([feat_1x1, feat_3x3, feat_5x5], dim=1)
        
        # Apply attention
        attention_weights = self.attention(multi_scale_features)
        attended_features = multi_scale_features * attention_weights
        
        # Global pooling and flatten
        x = self.global_pool(attended_features)
        x = x.view(x.size(0), -1)
        
        # Get embedding
        embedding = self.embedding(x)
        
        return embedding

class Pipeline:
    """
    End-to-end pipeline for logo processing, including downloading, feature extraction,
    and clustering.
    """
    def __init__(self, input_file, output_dir, max_workers=10, similarity_threshold=0.9):
        """
        Initialize the logo processing pipeline.
        
        Args:
            input_file (str): Path to the input file containing domain data
            output_dir (str): Directory to save output clusters
            max_workers (int): Maximum number of concurrent workers for threading. Default is 10.
            similarity_threshold (float): Threshold for considering logos similar. Default is 0.9.
        """
        self.max_workers = max_workers
        self.input_file = input_file
        self.output_dir = output_dir
        self.gemini_processed = 0
        self.model = None
        self.inference_transform = None
        self.df = None
        
        self.features = []
        self.clusters = []
        self.image_clusters = []
        self.unclustered = []
        
        self.similarity_threshold = similarity_threshold
        
    def load_data(self):
        """
        Load data from the input file into a pandas DataFrame.
        """
        # Load the CSV file into a DataFrame
        self.df = pd.read_parquet(self.input_file)
        self.df['extracted'] = False
        
        print("Data loaded successfully.")
        
        
    def process_domain(self, data):
        """
        Process a single domain to download its logo from Clearbit.
        
        Args:
            data (tuple): Tuple containing (index, domain)
            
        Returns:
            tuple: (index, success_status)
        """
        index, domain = data
        
        url = f"https://logo.clearbit.com/{domain}"
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Create images directory if it doesn't exist
                if not os.path.exists('images'):
                    os.makedirs('images')
                    
                with open(f"images/{index}.png", 'wb') as f:
                    f.write(response.content)
                return (index, True)
            else:
                return (index, False)
        except Exception as e:
            return (index, False)
        
    def download_images_clearbit(self):
        """
        Download logo images for all domains using the Clearbit API in parallel.
        """
        # Prepare the data for processing
        domains = list(enumerate(self.df['domain']))
        total = len(domains)

        # Initialize a progress bar
        progress_bar = tqdm(total=total, desc="Downloading logos", unit="logo")

        # Track successful downloads
        successful = 0

        # Using ThreadPoolExecutor for parallel requests
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks and get futures
            futures = [executor.submit(self.process_domain, data) for data in domains]
            
            # Process results as they complete
            for future in futures:
                try:
                    index, success = future.result()
                    if success:
                        self.df.at[index, 'extracted'] = True
                        successful += 1

                    progress_bar.set_postfix({"success": f"{successful}/{total}"})

                except Exception as e:
                    pass

                finally:
                    progress_bar.update(1)

        progress_bar.close()

        print(f"Downloaded {successful} logos successfully from Clearbit.")        

    
    def clean_html(self, html: str) -> str:
        """
        Extracts and cleans only the content inside the <header> tag by removing
        <script> comment tags.

        Args:
            html (str): Raw HTML content to clean.

        Returns:
            str: Cleaned content inside <header>, or an empty string if <header> is missing.
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')

            # Extract the header tag
            header = soup.header
            if not header:
                return ""  # Return empty string if no <header> exists

            # Remove <script> and <style> tags inside the header
            for tag in header(["script"]):
                tag.decompose()  # Removes tag and its content

            # Remove comments inside the header
            for comment in header.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

            # Return cleaned header HTML
            return header.prettify()

        except Exception:
            return html  # Return empty string if cleaning fails
        
    def download_png(self, url, save_path):
        """
        Download an image from a URL and save it to the specified path.
        
        Args:
            url (str): URL of the image to download
            save_path (str): Path where the image will be saved
            
        Returns:
            bool: True if download was successful
        """
        # Send GET request to the URL
        response = requests.get(url, stream=True)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Save the file
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                
        return True

    def proc_data(self, url, idx):
        """
        Process downloaded image data and save it with the appropriate extension.
        
        Args:
            url (str): URL of the image
            idx (int): Index for the saved file
        """
        if '.svg' in url:
            save_path = f'./images/{idx}.svg'
        elif '.png' in url:
            save_path = f'./images/{idx}.png'
        elif '.jpg' in url:
            save_path = f'./images/{idx}.jpg'
            
        self.download_png(url, save_path)

    def call_gemini(self, prompt):
        """
        Call the Gemini API with a prompt.
        
        Args:
            prompt (str): Prompt to send to Gemini API
            
        Returns:
            str: Response from Gemini API
        """
        # Configure the API key
        genai.configure(api_key=os.getenv("API_KEY"))
        
        # Generate content
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        return response.text

    def process_nth_domain(self, n):
        """
        Process the nth domain to extract its logo using website header and Gemini.
        
        Args:
            n (int): Index of the domain to process
            
        Returns:
            bool: True if logo extraction was successful
        """
        if self.df['extracted'][n] == False:
            try:
                url = f"https://www.{self.df['domain'][n]}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    html = response.text
                    header = self.clean_html(html)
                    
                    response = self.call_gemini(f""" Tell me what is the logo url for this header: {header}. 
                                            The domain is {self.df['domain'][n]}. 
                                            I want you to find only the url, no aditional information.
                                            The url should look like this: https://www.example.com/logo.png.
                                            If you can't find 'https' add it.
                                            If you can't find the logo url, just type 'NotFound'.
                                            
                                            It is MANDATORY that if a url is found it is returned with 'https://' in the beginning
                                            It is MANDATORY that if a url is not found the result is 'NotFound' exactly like this""")[:-1]
                                                  
                    if response.lower() == 'notfound':
                        return False
                    
                    self.proc_data(response, n)
                    
                    self.df.loc[n, 'extracted'] = True
                    
                    self.gemini_processed += 1
                    
                    return True
                    
            except Exception as e:
                return False
        else:
            return False

    def download_images_gemini(self):
        """
        Download logo images using Gemini AI to extract logo URLs from website headers.
        This is used as a fallback when Clearbit doesn't have the logo.
        """
        load_dotenv()
        
        total = len(self.df['domain'])
        successful = 0
        
        progress_bar = tqdm(total=total, desc="Downloading logos", unit="logo")
        # Using ThreadPoolExecutor for parallel requests
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks and get futures
            futures = [executor.submit(self.process_nth_domain, indx) for indx in range(total)]
            
            # Track successful downloads
            for future in futures:
                try:
                    if future.result():
                        successful += 1
                except Exception as e:
                    pass
                finally:
                    progress_bar.set_postfix({"success": f"{successful}"})
                    # Update progress bar
                    progress_bar.update(1)
                    
        progress_bar.close()

        print(f"Downloaded {successful} logos successfully with Gemini.")
    
    def clean_svg(self, svg_file_path):
        """
        Clean SVG file by replacing HTML entities with proper XML entities
        
        Args:
            svg_file_path (str): Path to the input SVG file
            
        Returns:
            str: Path to the cleaned SVG file, or None if cleaning failed
        """
        base, ext = os.path.splitext(svg_file_path)
        output_file_path = f"{base}_cleaned{ext}"
        
        try:
            # Read the SVG file
            with open(svg_file_path, 'r', encoding='utf-8') as file:
                svg_content = file.read()
            
            # Find and replace HTML entities
            def replace_entity(match):
                entity = match.group(1)
                # Convert HTML entity to its Unicode character
                return html.unescape(f"&{entity};")
            
            # Replace entities like &aacute; with their Unicode equivalents
            cleaned_content = re.sub(r'&([a-zA-Z]+);', replace_entity, svg_content)
            
            # Write the cleaned content to a new file
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_content)
            
            return output_file_path
            
        except Exception as e:
            print(f"Error cleaning SVG file {svg_file_path}: {e}")
            return None

    def convert_svgs(self):
        """
        Convert SVG files to PNG format using cairosvg for compatibility.
        """
                
        # Get all files in the images directory
        svgs = [file for file in os.listdir('./images') if file.endswith('.svg')]

        for svg_path in svgs:
            prefix = svg_path.split('.')[0]
            svg_path = f'./images/{prefix}.svg'
            png_path = f'./images/{prefix}.png'
            
            try:
                # Clean the SVG file first
                cleaned_svg = self.clean_svg(svg_path)
                
                if cleaned_svg:
                    # Convert the cleaned SVG to PNG
                    cairosvg.svg2png(url=cleaned_svg, write_to=png_path)
                    
                    # Remove the temporary cleaned file
                    os.remove(cleaned_svg)
                    
                    # Remove the original SVG files
                    os.remove(svg_path)
                    
            except Exception as e:
                os.remove(svg_path)
                os.remove(cleaned_svg)
        
        print("Converted SVGs to PNG successfully.")

    def load_model(self):
        """
        Initialize and load the LogoNet model for feature extraction.
        """
        # Initialize model
        self.model = LogoNet()
        self.model.eval()

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        # Example transform for inference
        self.inference_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def extract_logo_features(self, image_path):
        """
        Extract features from a logo image using the specialized logo model.
        
        Args:
            image_path (str): Path to the logo image
            
        Returns:
            tuple: (normalized_features, image_path) or None if extraction failed
        """
        try:
            # Read the image with alpha channel (if available)
            img = cv2.imread(f"./images/{image_path}", cv2.IMREAD_UNCHANGED)
            if img is None:
                return None

            # Check if the image has an alpha channel (transparency)
            if img.shape[-1] == 4:  # PNG with transparency
                bgr, alpha = img[:, :, :3], img[:, :, 3]  # Separate BGR and Alpha channels
                
                # Create a white background
                white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
                
                # Blend the image onto the white background
                alpha = alpha[:, :, np.newaxis] / 255.0  # Normalize alpha to range [0,1]
                img = (bgr * alpha + white_bg * (1 - alpha)).astype(np.uint8)  # Composite

            else:
                img = img[:, :, :3]  # Remove alpha channel if not needed

            img = Image.fromarray(img)
            
            # Apply transformations
            image = self.inference_transform(img).unsqueeze(0)
            
            # Move to the same device as model
            device = next(self.model.parameters()).device
            image = image.to(device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image)
            
            # Return normalized features
            return F.normalize(features, p=2, dim=1).view(1, -1), image_path
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
        
    def extract_features(self):
        """
        Extract features from all downloaded logo images using the model.
        """
        filenames = os.listdir('./images')
        print(f"Extracting features from {len(filenames)} images...")

        total = len(filenames)

        progress_bar = tqdm(total=total, desc="Extracting features", unit="logo")
        
        # Using ThreadPoolExecutor for parallel requests
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks and get futures
            futures = [executor.submit(self.extract_logo_features, path) for path in filenames]
            
            # Track successful downloads
            for future in futures:
                # Update progress bar
                progress_bar.update(1)

                self.features.append(future.result())
                    
        progress_bar.close()
        
        self.features = [feature for feature in self.features if feature is not None]
        
        print(f"Extracted {len(self.features)} features successfully.")

    def generate_clusters(self):
        """
        Generate clusters of similar logos based on feature similarity.
        Uses BallTree for efficient nearest neighbor search.
        """
        # During initialization
        features_array = np.vstack([f[0].cpu().numpy() for f in self.features if f is not None])
        filenames = [f[1] for f in self.features if f is not None]
        
        normalized_features = features_array / np.linalg.norm(features_array, axis=1, keepdims=True)
        self.tree = BallTree(normalized_features, leaf_size=40, metric='euclidean')

        # During clustering
        clusters = {}
        indices_map = {}
        for i, feature in enumerate(tqdm(normalized_features, desc="Clustering", unit="logo")):
            # Find all neighbors within threshold distance
            indices = self.tree.query_radius([feature], r=1-self.similarity_threshold)[0]
            
            # Find which existing cluster this belongs to, if any
            if indices.size > 0:
                # Filter indices to only include those that have already been assigned to clusters
                assigned_indices = [i for i in indices if i in indices_map]
                
                if assigned_indices:  # If we have any assigned neighbors
                    clusters_found = [indices_map[i] for i in assigned_indices]
                    counter = Counter(clusters_found)
                    most_common = counter.most_common(1)
                    clusters[most_common[0][0]].append(i)
                    indices_map[i] = most_common[0][0]
                else:
                    # No previously assigned neighbors, create new cluster
                    new_id = len(clusters)
                    clusters[new_id] = [i]
                    indices_map[i] = new_id
            else:
                # No similar points found, create a new cluster
                new_id = len(clusters)
                clusters[new_id] = [i]
                indices_map[i] = new_id
                
        # Convert clusters to list of lists
        self.clusters = [clusters[i] for i in clusters]
        self.image_clusters = [[filenames[i] for i in cluster] for cluster in self.clusters]
        
        self.unclustered = [cluster[0] for cluster in self.image_clusters if len(cluster) == 1]
        self.image_clusters = [cluster for cluster in self.image_clusters if len(cluster) > 1] 
        
        print("Clustering completed.")
        print(f"Number of clusters: {len(self.image_clusters)}")

    def generate_output(self):
        """
        Generate output directories with clustered logos and save cluster information to JSON.
        """
        
        for i in range(len(self.image_clusters)):
            if not os.path.exists(f"./clusters/cluster{i}"):
                os.makedirs(f"./clusters/cluster{i}")

            for image_path in self.image_clusters[i]:
                os.rename(f"./images/{image_path}", f"./clusters/cluster{i}/{image_path}")
                
        if not os.path.exists(f"./clusters/a_unclustered"):
            os.makedirs(f"./clusters/a_unclustered")
            
        for image_path in self.unclustered:
            os.rename(f"./images/{image_path}", f"./clusters/a_unclustered/{image_path}")
                
        domain_clusters = [[self.df['domain'][int(y.split('.')[0])] for y in x]for x in self.image_clusters]
        json.dump(domain_clusters, open('clusters.json', 'w'), indent=4)
        
        shutuil.rmtree('./images')
        print("Output generated successfully.")

    def run(self, run_with_gemini=False):
        """
        Run the complete pipeline from data loading to output generation.
        
        Args:
            run_with_gemini (bool): Whether to use Gemini API as fallback. Default is False.
        """
        
        self.load_data()
        
        self.download_images_clearbit()
        
        if run_with_gemini:
            self.download_images_gemini()
        
            self.convert_svgs()
        
        self.load_model()
        
        self.extract_features()

        self.generate_clusters()        

        self.generate_output()
        
        print("Pipeline completed successfully.")
        
if __name__ == "__main__":
    input_file = input("Enter the path to the input .parquet file : ")
    output_dir = 'clusters'
    pipeline = Pipeline(input_file, output_dir, similarity_threshold=0.6)
    
    run_with_gemini = len(sys.argv) > 1 and sys.argv[1] == 'run_with_gemini'
    
    if run_with_gemini:
        print("Using Gemini API for logo extraction.")

    pipeline.run(run_with_gemini)