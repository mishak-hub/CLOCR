import os
import pandas as pd
import argparse

def observe_ocr_paper_correction_results(sheet_names: list[str]):
  """This function just shows the results for the four models trained and tested by the llms_post-ocr_correction project's team. Taken from their results.ipynb file.
  """
  results = {}
  for sheet in sheet_names:
    results[sheet] = pd.read_csv(f'results/{sheet}.csv')

    corrections = results[sheet]
    print("****************************************************\n\n", sheet, "\n")
    print(corrections.head(10))
    
    for i in range(3): # len(corrections)
      print(i+1)
      print(f"OCR Text:\n{corrections['OCR Text'][i]}\n")
      print(f"Ground Truth:\n{corrections['Ground Truth'][i]}\n")
      print(f"Model Correction:\n{corrections['Model Correction'][i]}\n\n\n\n")
      
    print(corrections.describe())
    
if __name__ == '__main__':
  # Parse arguments for model/config/data
  parser = argparse.ArgumentParser(description='Observing LM test results')
  parser.add_argument('sheet_names', nargs='+', type=str, help='List of CSV files in results folder to analyze')
  args = parser.parse_args()
  
  print(args.sheet_names)
  
  observe_ocr_paper_correction_results(args.sheet_names)