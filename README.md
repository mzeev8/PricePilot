# Price Prediction Using Large Language Models

This project builds an end-to-end pipeline for **price prediction using large language models (LLMs)** by curating structured product metadata into high-quality training prompts. The system processes large-scale Amazon product datasets, cleans and filters the data, constructs supervised prompts, and evaluates model performance through training and testing notebooks.

The project is designed for experimentation with modern transformer-based models and focuses on **data quality, prompt construction, and evaluation** rather than UI or deployment.

---

## Project Overview

The core idea is to predict the **price of a product to the nearest dollar** given its textual metadata (title, description, features, and details). Each datapoint is converted into a structured prompt suitable for supervised fine-tuning or evaluation of an LLM.

---

## Key Components

- Dataset loading from large-scale Amazon metadata
- Aggressive text cleaning and normalization
- Token-length–aware filtering for LLM compatibility
- Prompt construction for supervised learning
- Baseline testing and evaluation notebooks
- Modular, reusable data-loading pipeline

---

## Project Structure

```

.
├── Training.ipynb                      # Model training experiments
├── Testing.ipynb                       # Evaluation and inference
├── baseline_testing.ipynb              # Baseline model performance
├── Dissertation_Curating_Dataset.ipynb # Dataset curation and analysis
├── items.py                            # Item abstraction and prompt construction
├── loaders.py                          # Parallel dataset loading and filtering
└── README.md

```

---

## Data Pipeline

### Item Abstraction (`items.py`)
Each product is converted into an `Item` object that:
- Cleans and normalizes raw product text
- Removes noisy or irrelevant metadata
- Enforces minimum and maximum token limits
- Generates a supervised training prompt of the form:

```

How much does this cost to the nearest dollar?

<product text>

Price is $XX.00

```

Only datapoints that meet strict quality and token constraints are included.

---

### Dataset Loader (`loaders.py`)
The loader:
- Pulls data from the Amazon Reviews 2023 dataset
- Filters products by valid price range
- Processes data in chunks for scalability
- Uses multiprocessing to accelerate preprocessing
- Outputs curated, category-labeled `Item` objects

---

## Notebooks

### Training.ipynb
- Loads curated prompts
- Trains or fine-tunes a language model
- Tracks token counts and dataset statistics

### Testing.ipynb
- Evaluates trained models
- Compares predicted vs actual prices
- Analyzes error patterns

### baseline_testing.ipynb
- Establishes baseline performance
- Useful for benchmarking improvements

### Dissertation_Curating_Dataset.ipynb
- Exploratory analysis
- Dataset validation
- Documentation of design choices

---

## Dependencies

Key libraries used:
- transformers
- datasets
- tqdm
- concurrent.futures
- NumPy
- Python 3.9+

Install dependencies as needed using pip.

---

## Use Cases

- LLM supervised fine-tuning
- Prompt engineering research
- Data curation for price prediction
- Academic or dissertation-level experimentation

---

## Future Improvements

- Support for multi-currency prediction
- Regression-style decoding instead of classification
- Instruction-tuned model comparison
- Prompt format ablation studies
- Dataset balancing by category

---

## Author

Mohammed Zaid V
