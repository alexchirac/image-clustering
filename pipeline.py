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
from google import genai 
from dotenv import load_dotenv
import shutil as shutuil
import cv2
import numpy as np


class LogoNet(nn.Module):
    """
    A specialized neural network for logo recognition with:
    1. Enhanced local feature extraction
    2. Multi-scale processing
    3. Shape-aware attention
    4. Rotation and scale invariance
    """
    def __init__(self, embedding_dim=512):
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
    def __init__(self, input_file, output_dir, max_workers=10, similarity_threshold=0.9):
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
        
        self.similarity_threshold = similarity_threshold
        
    def load_data(self):
        # Load the CSV file into a DataFrame
        self.df = pd.read_parquet(self.input_file)
        self.df['extracted'] = False
        
        print("Data loaded successfully.")
        
        
    def process_domain(self, data):
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
        if '.svg' in url:
            save_path = f'./images/{idx}.svg'
        elif '.png' in url:
            save_path = f'./images/{idx}.png'
        elif '.jpg' in url:
            save_path = f'./images/{idx}.jpg'
            
        self.download_png(url, save_path)

    def call_gemini(self, prompt):
        client = genai.Client(api_key=os.getenv("API_KEY"))

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        return response.text

    def process_nth_domain(self, n):
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
        
        Parameters:
        svg_file_path (str): Path to the input SVG file
        output_file_path (str): Path where the cleaned SVG will be saved (optional)
        
        Returns:
        str: Path to the cleaned SVG file
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
        Convert SVG files to PNG format using cairosvg.
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
        
    # Function to extract logo features using pretrained model
    def extract_logo_features(self, image_path):
        """Extract features from a logo image using a specialized logo model"""
        try:
            # # Open image
            # image = Image.open(f"./images/{image_path}").convert('RGB')
            
             
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
            
            # # Convert to grayscale
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
        # Total number of features to process
        total_features = len(self.features)

        # Iterate through features with a progress bar
        for feature in tqdm(self.features, desc="Clustering Images", unit="image", total=total_features):
            if feature is None:
                continue
            feature, filename = feature
            
            best_cluster = -1
            best_cluster_raport = 0

            for i in range(len(self.clusters)):
                cluster = self.clusters[i]
                close = 0
                far = 0

                for image_feature in cluster:
                    similarity = F.cosine_similarity(feature, image_feature, dim=1).item()
                    
                    if similarity < self.similarity_threshold:
                        far += 1
                    else:
                        close += 1

                if close / (close + far) > 0.8 and close / (close + far) > best_cluster_raport:
                    best_cluster = i
                    best_cluster_raport = close / (close + far)

            if best_cluster == -1:
                self.clusters.append([feature])
                self.image_clusters.append([filename])
            else:
                self.clusters[best_cluster].append(feature)
                self.image_clusters[best_cluster].append(filename)
                
        print("Clustering completed.")
        print(f"Number of clusters: {len(self.clusters)}")

    def generate_output(self):
        
        for i in range(len(self.image_clusters)):
            if not os.path.exists(f"./clusters/cluster{i}"):
                os.makedirs(f"./clusters/cluster{i}")

            for image_path in self.image_clusters[i]:
                os.rename(f"./images/{image_path}", f"./clusters/cluster{i}/{image_path}")
                
        domain_clusters = [[self.df['domain'][int(y.split('.')[0])] for y in x]for x in self.image_clusters]
        json.dump(domain_clusters, open('clusters.json', 'w'), indent=4)
        
        shutuil.rmtree('./images')
        print("Output generated successfully.")

    def run(self):
        
        self.load_data()
        
        self.download_images_clearbit()
        
        # self.download_images_gemini()
        
        # self.convert_svgs()
        
        self.load_model()
        
        self.extract_features()

        self.generate_clusters()        

        self.generate_output()
        
        print("Pipeline completed successfully.")
        
if __name__ == "__main__":
    input_file = 'logos.snappy.parquet'
    output_dir = 'clusters'
    pipeline = Pipeline(input_file, output_dir, similarity_threshold=0.8)
    pipeline.run()
    