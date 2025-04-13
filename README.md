# Logo Processing Pipeline

A scalable solution for extracting, processing, and clustering company logos based on visual similarity.

## Overview

This project is a comprehensive logo clustering system that takes domains from different companies and groups their logos based on visual similarity. It employs advanced feature extraction using a neural network based on EfficientNet-B5 and clusters them using a BallTree algorithm to identify logos that likely belong to the same entity or brand family.

## Key Features

- Multi-source logo extraction (Clearbit API and Gemini AI)
- Advanced deep learning-based feature extraction
- Efficient nearest-neighbor clustering using BallTree
- SVG to PNG conversion for unified processing
- Highly parallelizable architecture for scalability

## Requirements

- Python 3.8 or higher
- Pip package manager
- Virtual environment (recommended)

## Installation

1. Clone the repository or download the pipeline code
2. Navigate to the project directory

## How to Run

To run the script, follow these steps:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Options

**With Gemini AI (recommended for better logo extraction)**:

If you have a Gemini API key, create a `.env` file in the project root with:
```
API_KEY=[your-key]
```

Then run:
```bash
python3 pipeline.py run_with_gemini
```

**Without Gemini AI**:

To run without using the Gemini API for logo extraction:
```bash
python3 pipeline.py
```

## Output

After running, the script will create a `clusters` directory containing:
- Subdirectories for each cluster of similar logos
- A special `a_unclustered` directory for logos that don't match any others
- A `clusters.json` file mapping clusters to domain names

## Technical Details

### Logo Extraction Process

The pipeline employs a multi-stage approach to maximize logo extraction success rates:

1. **Clearbit API**: The primary source for logo extraction, providing approximately 82% coverage.
2. **Gemini AI**: Used as a secondary source to extract logos from website headers, boosting the extraction rate to approximately 91%.

> **⚠️ Important Note**: Clearbit API is scheduled to be discontinued on December 1, 2025. An alternative solution will need to be implemented before then.

### Feature Extraction Evolution

The feature extraction methodology went through several iterations:

1. **Image Histograms**: Initially attempted but proved inadequate for capturing logo similarities.
2. **ResNet Models**: Tested but tended to cluster based on semantic meaning rather than visual appearance.
3. **EfficientNet Series**: Progressively tested from B0 to B5, with B5 delivering the best results for balancing detail capture and computational efficiency.

The final implementation uses a custom `LogoNet` architecture based on EfficientNet-B5 with specialized attention mechanisms for logo-specific features.

### Clustering Approach

For efficient clustering without traditional ML clustering algorithms:

1. **Initial Brute Force**: A comparative approach testing all logos against each other (not scalable).
2. **BallTree Implementation**: Significantly improved efficiency by using a BallTree data structure to find nearest neighbors within a similarity threshold.
3. **Cluster Assignment**: Logos are assigned to clusters based on the highest number of similar neighbors, ensuring related logos are grouped together.

## Scalability

The solution is designed for high scalability:

- Logo extraction processes (both Clearbit and Gemini) are implemented with ThreadPoolExecutor for parallel processing
- Feature extraction is parallelizable across multiple workers
- The BallTree approach provides O(log n) lookup complexity compared to O(n²) in brute force methods
- All major components support distributed computing approaches for further scaling

## Future Improvements

Several opportunities for enhancement exist:

1. **Alternative Logo Sources**: As Clearbit will be deprecated, implementing additional logo extraction methods from sources like company registries or specialized APIs
2. **Incremental Processing**: Adding support for incremental updates to avoid reprocessing the entire dataset
3. **Hierarchical Clustering**: Implementing hierarchical clustering to discover relationships between logo clusters
4. **Confidence Scoring**: Adding confidence scores for cluster assignments to facilitate manual verification of uncertain cases

## License

[Apache 2.0]