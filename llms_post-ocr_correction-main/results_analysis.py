import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Levenshtein import distance as levenshtein_distance
import torch



# Metric calculation functions
def compute_normalized_edit_distance(text1, text2):
    """Calculate normalized edit distance."""
    edit_distance = levenshtein_distance(str(text1), str(text2))
    max_length = max(len(text1), len(text2))
    return 1 - (edit_distance / max_length) if max_length > 0 else 1

def compute_normalized_cosine_similarity(text1, text2, model_name="paraphrase-MPNet-base-v2"):
    """Calculate normalized cosine similarity."""
    model = SentenceTransformer(model_name, device = 'cuda' if torch.cuda.is_available() else 'cpu')
    embeddings = model.encode([text1, text2], batch_size=64, show_progress_bar=True)  # Example with smaller batch size
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return (similarity + 1) / 2  # Normalize to [0, 1]

def compute_combined_metric(text1, text2, alpha=0.5):
    """Calculate combined metric."""
    normalized_cosine = compute_normalized_cosine_similarity(text1, text2)
    normalized_edit = compute_normalized_edit_distance(text1, text2)
    return alpha * normalized_cosine  + (1 - alpha) * normalized_edit

# Observing results and calculating metrics
def observe_ocr_paper_correction_results(results_folder, alpha=0.5, output_file="combined_results.csv"):
    """Automatically detects CSV files, calculates metrics, and saves the results."""
    # Automatically detect all CSV files in the folder
    sheet_names = [f.replace('.csv', '') for f in os.listdir(results_folder) if f.endswith('.csv')]
    print(f"Detected sheet names: {sheet_names}")

    results = []
    for sheet in sheet_names[:1]:
        file_path = os.path.join(results_folder, f"{sheet}.csv")
        corrections = pd.read_csv(file_path)
        print("****************************************************\n\n", sheet, "\n")
        print(corrections.head(10))
        
        # Check required columns
        if not all(col in corrections.columns for col in ["OCR Text", "Ground Truth", "Model Correction"]):
            print(f"Skipping {sheet}: Required columns missing.")
            continue
        
        # Process each row
        for i, row in corrections.iterrows():
            ocr_text = row["OCR Text"]
            ground_truth = row["Ground Truth"]
            model_correction = row["Model Correction"]

            # Print first 3 examples for observation
            if i < 20:
                print(f"{i+1}. OCR Text:\n{ocr_text}\n")
                print(f"Ground Truth:\n{ground_truth}\n")
                print(f"Model Correction:\n{model_correction}\n\n")

            # Calculate metrics
            metric_0 = compute_combined_metric(ocr_text, ground_truth, alpha)
            metric_1 = compute_combined_metric(model_correction, ground_truth, alpha)
            metric_prime = compute_combined_metric(ocr_text, model_correction, alpha)

            # Append results
            results.append({
                "File Name": sheet,
                "OCR Text": ocr_text,
                "Ground Truth": ground_truth,
                "Model Correction": model_correction,
                "Metric 0 (OCR vs Ground Truth)": metric_0,
                "Metric 1 (Model Correction vs Ground Truth)": metric_1,
                "Metric ' (OCR vs Model Correction)": metric_prime
            })
        
        # Print summary stats for the sheet
        print(corrections.describe())
    
    # Save combined metrics to a new CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"\nCombined results saved to: {output_file}")

# Main script
if __name__ == '__main__':
    print(torch.cuda.is_available())
    # Default results folder and output file
    results_folder = "results"  # Replace with your folder path
    output_csv = "combined_results.csv"
    alpha = 0.5  # Adjust alpha as needed

    print(f"Results folder: {results_folder}")
    print(f"Output file: {output_csv}")

    observe_ocr_paper_correction_results(results_folder, alpha=alpha, output_file=output_csv)