from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Levenshtein import distance as levenshtein_distance
import torch
from torch.amp import autocast
import pickle  # For saving and loading embeddings

# Metric calculation functions
def compute_normalized_edit_distance(text1, text2):
    """Calculate normalized edit distance."""
    edit_distance = levenshtein_distance(str(text1), str(text2))
    max_length = max(len(text1), len(text2))
    return 1 - (edit_distance / max_length) if max_length > 0 else 1

def compute_normalized_cosine_similarity(text1, text2, model_name="paraphrase-MPNet-base-v2", cache_dir="embeddings_cache"):
    """Calculate normalized cosine similarity with cached embeddings."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)

    # Define a unique cache filename based on the texts
    cache_filename = f"{hash(text1)}_{hash(text2)}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)

    # Check if the embeddings are cached
    if os.path.exists(cache_path):
        # Load cached embeddings
        with open(cache_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"L: {text1[:10]}... and {text2[:10]}...")  # Load message (L)
    else:
        # Compute and cache the embeddings if not already cached
        print(f"C: {text1[:10]}... and {text2[:10]}...")  # Compute message (C)
        processed_texts = [text1, text2]

        # Use mixed precision (16-bit) for faster processing
        with autocast("cuda"):  # Mixed precision
            embeddings = model.encode(processed_texts, batch_size=64, show_progress_bar=False)

        # Save the embeddings to the cache file
        os.makedirs(cache_dir, exist_ok=True)  # Create the cache directory if it doesn't exist
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)

    # Empty GPU memory cache to ensure unused memory is freed up
    torch.cuda.empty_cache()

    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return (similarity + 1) / 2  # Normalize to [0, 1]

def compute_combined_metric(text1, text2, alpha=0.5):
    """Calculate combined metric."""
    normalized_cosine = compute_normalized_cosine_similarity(text1, text2)
    normalized_edit = compute_normalized_edit_distance(text1, text2)
    return alpha * normalized_cosine  + (1 - alpha) * normalized_edit

# Function to calculate the metrics for each row
def process_row(row, alpha, cache_dir="embeddings_cache"):
    ocr_text = row["OCR Text"]
    ground_truth = row["Ground Truth"]
    model_correction = row["Model Correction"]

    # Calculate metrics
    metric_0 = compute_combined_metric(ocr_text, ground_truth, alpha)
    metric_1 = compute_combined_metric(model_correction, ground_truth, alpha)
    metric_prime = compute_combined_metric(ocr_text, model_correction, alpha)

    return {
        "OCR Text": ocr_text,
        "Ground Truth": ground_truth,
        "Model Correction": model_correction,
        "Metric 0 (OCR vs Ground Truth)": metric_0,
        "Metric 1 (Model Correction vs Ground Truth)": metric_1,
        "Metric ' (OCR vs Model Correction)": metric_prime
    }

# Observing results and calculating metrics
def observe_ocr_paper_correction_results(results_folder, alpha=0.5, output_file="combined_results.csv", cache_dir="embeddings_cache"):
    """Automatically detects CSV files, calculates metrics, and saves the results."""
    # Automatically detect all CSV files in the folder
    sheet_names = [f.replace('.csv', '') for f in os.listdir(results_folder) if f.endswith('.csv')]
    print(f"Detected sheet names: {sheet_names}")

    results = []
    for sheet in sheet_names[:1]:  # Process only the first sheet for observation
        file_path = os.path.join(results_folder, f"{sheet}.csv")
        corrections = pd.read_csv(file_path)
        print("****************************************************\n\n", sheet, "\n")
        print(corrections.head(10))

        # Check required columns
        if not all(col in corrections.columns for col in ["OCR Text", "Ground Truth", "Model Correction"]):
            print(f"Skipping {sheet}: Required columns missing.")
            continue

        # Use ThreadPoolExecutor to parallelize the processing of rows
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_row, row, alpha, cache_dir) for _, row in corrections.iterrows()]
            for future in futures:
                results.append(future.result())

        # Print summary stats for the sheet
        print(corrections.describe())

    # Save combined metrics to a new CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"\nCombined results saved to: {output_file}")

# Main script
if __name__ == '__main__':
    print(torch.cuda.is_available())
    
    # Set the memory limit to use only 80% of the available GPU memory
    torch.cuda.set_per_process_memory_fraction(0.8, device=0)

    # Default results folder and output file
    results_folder = "results"  # Replace with your folder path
    output_csv = "combined_results.csv"
    alpha = 0.5  # Adjust alpha as needed

    print(f"Results folder: {results_folder}")
    print(f"Output file: {output_csv}")

    observe_ocr_paper_correction_results(results_folder, alpha=alpha, output_file=output_csv)
