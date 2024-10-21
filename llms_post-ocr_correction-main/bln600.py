# This file is a version of bln600.ipynb which can be
# invoked on Newton to prepare the dataset CSV.

from sklearn.model_selection import train_test_split
import json
import Levenshtein
import os
import pandas as pd

# Compute character error rate (CER)
def cer(prediction, target):
    distance = Levenshtein.distance(prediction, target)
    return distance / len(target)

# Helper function to preprocess text
def preprocess(c):
    c = c.str.replace("‘", "'", regex=False)
    c = c.str.replace("’", "'", regex=False)
    c = c.str.replace("“", '"', regex=False)
    c = c.str.replace("”", '"', regex=False)
    c = c.str.replace("—", "-", regex=False)
    c = c.str.replace(r'\s+', ' ', regex=True)
    c = c.str.strip()
    return c
  
sample_id, date, publication, ocr, gt = [], [], [], [], []

with open('data/BLN600/metadata.json', 'r') as f:
    metadata = json.load(f)

for doc in metadata:
    sample_id.append(doc['short_id'])
    
    d = doc['date']
    d = f'{d[:4]}-{d[4:6]}-{d[6:]}'
    date.append(d)
    
    p = doc['publication']
    match doc['publication']:
            case 'Lloyd&apos;s Weekly London Newspaper':
                p = 'Lloyd\'s Illustrated Newspaper'
            case 'Lloyd&apos;s Weekly Newspaper':
                p = 'Lloyd\'s Illustrated Newspaper'
            case 'The Illustrated Police News etc':
                p = 'Illustrated Police News'
            case 'The Morning Chronicle':
                p = 'Morning Chronicle'
            case 'The Era':
                p = 'The Era'
            case 'The Charter':
                p = 'Charter'
            case 'Daily News':
                p = 'Daily News'
    publication.append(p)    
    
    f = open(os.path.join('data/BLN600/OCR Text', doc['short_id'] + '.txt'), 'r')
    ocr.append(' '.join(f.read().split()))
    f.close()
    
    f = open(os.path.join('data/BLN600/Ground Truth', doc['short_id'] + '.txt'), 'r')
    gt.append(' '.join(f.read().split()))
    f.close()

bln600 = pd.DataFrame({'Sample ID': sample_id, 'Date': date, 'Publication': publication, 'OCR Text': ocr, 'Ground Truth': gt})
bln600['Date'] = pd.to_datetime(bln600['Date'])
# bln600.head(10)

# The above dataframe could be saved as a CSV, but for 
# token limit and training dataset size purposes, the 
# following is superior:

sample_id, date, publication, ocr, gt = [], [], [], [], []

for s in bln600['Sample ID']:
    with open(os.path.join('data/Sequences', f'{s}.txt'), 'r') as f:
        lines = f.readlines()
    
    ocr_text, ground_truth = '', ''
    for line in lines:
        if line.startswith('OCR Text: '):
            ocr_text = line.replace('OCR Text: ', '').strip()
        elif line.startswith('Ground Truth: '):
            ground_truth = line.replace('Ground Truth: ', '').strip()
        if ocr_text and ground_truth:
            sample_id.append(s)
            date.append(bln600.loc[bln600['Sample ID'] == s, 'Date'].iloc[0])
            publication.append(bln600.loc[bln600['Sample ID'] == s, 'Publication'].iloc[0])
            ocr.append(ocr_text)
            gt.append(ground_truth)            
            ocr_text, ground_truth = '', ''

seq = pd.DataFrame({'Sample ID': sample_id, 'Date': date, 'Publication': publication, 'OCR Text': ocr, 'Ground Truth': gt})
seq['OCR Text'] = preprocess(seq['OCR Text'])
seq['Ground Truth'] = preprocess(seq['Ground Truth'])
seq['CER'] = seq.apply(lambda row: cer(row['OCR Text'], row['Ground Truth']), axis=1)
seq.to_csv('datasets/bln600.csv', index=False)
# seq.head(10)
# seq['CER'].describe()


# # Split into train/test/val sets
# train_ids, test_ids = train_test_split(seq['Sample ID'].unique(), test_size=0.2, random_state=600)
# # train_ids, val_ids = train_test_split(train_ids, test_size=0.125, random_state=600)

# train = seq[seq['Sample ID'].isin(train_ids)]
# # val = seq[seq['Sample ID'].isin(val_ids)]
# test = seq[seq['Sample ID'].isin(test_ids)]
# train_ids, test_ids = train_test_split(seq['Sample ID'].unique(), test_size=0.2, random_state=600)
# # train_ids, val_ids = train_test_split(train_ids, test_size=0.125, random_state=600)

# train = seq[seq['Sample ID'].isin(train_ids)]
# # val = seq[seq['Sample ID'].isin(val_ids)]
# test = seq[seq['Sample ID'].isin(test_ids)]

# train.to_csv('data/train.csv', index=False)
# test.to_csv('data/test.csv', index=False)
