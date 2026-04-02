# IMDB.drama Clustering Project

## English

### Overview
This project performs clustering analysis on the IMDB.drama dataset from OpenML.

It includes:
- data download
- sparse data inspection
- preprocessing
- clustering with multiple algorithms
- evaluation and comparison
- 2D and 3D visualization

---

### Setup

python3 -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  

---

### Execution steps

1. Download data

python -m scripts.download_data

2. Inspect sparse dataset (optional)

python -m scripts.inspect_sparse_data

3. Preprocessing

python -m scripts.run_preprocessing

Outputs:
- data/processed/
- models/preprocessing/

4. Run clustering experiments

python -m scripts.run_experiments

Outputs:
- results/metrics/clustering_metrics.csv
- results/cluster_reports/
- models/clustering/
- results/configs/

5. Generate visualizations

python -m scripts.run_visualizations

Outputs:
- results/plots/

---

### Algorithms used
- MiniBatchKMeans
- Agglomerative Clustering
- Gaussian Mixture Model
- DBSCAN
- BIRCH

---

### Notes on reproducibility
- All random states are fixed
- Pipeline must be run in order
- Results can be regenerated from raw data

---

## Srpski

### Opis
Projekat vrši klasterovanje IMDB.drama skupa podataka sa OpenML.

Obuhvata:
- preuzimanje podataka
- analizu sparse strukture
- preprocesiranje
- primenu više algoritama klasterovanja
- evaluaciju rezultata
- 2D i 3D vizuelizaciju

---

### Pokretanje

python3 -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  

---

### Koraci

1. Preuzimanje podataka

python -m scripts.download_data

2. Provera podataka (opciono)

python -m scripts.inspect_sparse_data

3. Preprocesiranje

python -m scripts.run_preprocessing

4. Klasterovanje

python -m scripts.run_experiments

5. Vizuelizacija

python -m scripts.run_visualizations

---

### Napomena
- Random seed je fiksiran
- Skripte se pokreću redom
- Rezultati su reproduktivni