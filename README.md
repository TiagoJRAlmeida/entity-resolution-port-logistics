# entity-resolution-port-logistics

https://github.com/user-attachments/assets/4af17b18-dea2-4252-b253-a41047bacbf1

Data standardization and entity resolution system for large-scale port logistics records, developed in collaboration with the Port of Sines under the NEXUS program.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Option A: Docker](#option-a-docker)
  - [Option B: Python Virtual Environment](#option-b-python-virtual-environment)
- [Usage](#usage)
- [Testing the Algorithm](#testing-the-algorithm)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

---

## Overview

Companies operating in large logistics environments tend to accumulate inconsistent records over time — the same entity appearing under dozens of name variants due to manual data entry, abbreviations, typos, and formatting differences. At scale, this makes statistical analysis unreliable and cross-dataset reconciliation impractical.

This project was developed as the capstone of a Computer Engineering degree at the University of Aveiro, in partnership with APS — Administração dos Portos de Sines e do Algarve, S.A. The system processes datasets with ~4 million rows and ~60 columns, focusing on the `name` and `Identification_number` fields.

The approach combines semantic embeddings (SentenceTransformers), approximate nearest-neighbour search (FAISS), and post-processing filters (Levenshtein distance, Jaccard similarity) to cluster name variants and assign a canonical representative to each cluster. The final output is a synonym map that can be used to standardise any downstream dataset.

**Key metrics (ST+FAISS approach):**

| Metric    | Without reference table | With 50% reference table |
|-----------|------------------------|--------------------------|
| Precision | 0.87                   | 0.85                     |
| Recall    | 0.75                   | 0.82                     |
| F1-Score  | 0.80                   | 0.83                     |

---

## System Architecture

The pipeline runs in seven stages:

1. **Preprocessing** — Lowercase conversion, punctuation removal, stripping of legal suffixes (e.g. "Lda.", "S.A.") and geographic terms.
2. **Vectorisation** — Semantic embeddings generated per company name via SentenceTransformers.
3. **Clustering** — FAISS-indexed approximate nearest-neighbour search groups semantically similar names. A configurable similarity threshold (default: 0.81) controls cluster tightness.
4. **Post-processing** — Noisy clusters are refined using a weighted combination of normalised Levenshtein distance (70%) and Jaccard similarity (30%). Outliers are isolated into their own groups.
5. **Country prefix separation** — Clusters containing names from multiple countries (inferred from suffixes like PT, BR, ES) are subdivided accordingly.
6. **Canonical name assignment** — Each cluster is assigned the name closest to the cluster centroid as its canonical representative.
7. **Synonym map export** — A final JSON map (`name_variant → canonical_name`) is generated and used to update the input datasets.

An optional reference table (ground truth) can be supplied to seed initial clusters and improve F1-score from 0.80 to 0.83.

---

## Project Structure

```
.
├── Dockerfile
├── README.md
├── requirements.txt
├── report/
│   ├── report.pdf
│   ├── report.tex
│   ├── referencias.bib
│   └── images/
├── sample_data/
│   └── sample_companies.csv      # Sample dataset for testing
└── src/
    ├── main.py                    # Entry point
    ├── interface.py               # Terminal UI
    ├── string_cleaning.py         # Preprocessing logic
    ├── requirements.txt
    └── clustering/
        ├── clustering.py          # Core clustering algorithm
        └── accuracy_test.py       # Evaluation metrics
```

---

## Getting Started

### Option A: Docker

> Requires [Docker](https://www.docker.com/) installed.

```bash
# Build the image
docker build -t entity-resolution .

# Run with your data mounted
docker run -it \
  -v $(pwd)/sample_data:/app/data \
  entity-resolution
```

The container will start the TUI and prompt you to select your input files from `/app/data`.

### Option B: Python Virtual Environment

> Requires Python 3.9+.

```bash
# Clone the repository
git clone https://github.com/TiagoJRAlmeida/entity-resolution-port-logistics.git
cd entity-resolution-port-logistics

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r src/requirements.txt

# Run the application
cd src
python3 main.py
```

---

## Usage

Once running, the TUI will guide you through the following steps:

1. **Select input file(s)** — One or more CSV/Excel datasets to be corrected. Each must contain at least a `name` column and an `Identification_number` column.
2. **Select reference table (optional)** — A pre-validated dataset used to seed the ground truth clusters. Providing this improves output quality.
3. **Choose evaluation mode** — Optionally enable accuracy reporting (precision, recall, F1-score) if a reference table was supplied.
4. **Wait for processing** — Progress is displayed live in the terminal. On a standard laptop, a ~4M row dataset completes in approximately 20 minutes.
5. **Collect output** — The corrected dataset and synonym map are written to the working directory.

---

## Testing the Algorithm

A sample dataset is included in `sample_data/sample_companies.csv` to allow testing without access to the original Port of Sines data (which is confidential).

The file contains ~90 records representing ~15 distinct companies, each appearing under 5–6 name variants with realistic inconsistencies: capitalisation differences, missing accents, abbreviated legal suffixes, and minor typos. It also includes records from Portuguese (PT), Brazilian (BR), and Spanish (ES) entities to exercise the country prefix separation step.

```bash
# From the src/ directory, with the virtual environment active:
python3 main.py
# When prompted, select: ../sample_data/sample_companies.csv
```

To run the accuracy evaluation directly:

```bash
cd src
python3 -m clustering.accuracy_test
```

---

## Results

The final system (SentenceTransformers + FAISS) achieved an F1-score of ~0.80 without a reference table and ~0.83 with one, processing ~4 million records in approximately 20 minutes on a consumer laptop — compared to 1h30 for the initial MinHash prototype.

See [`report/report.pdf`](report/report.pdf) for the full technical report, including algorithm design, evaluation methodology, and a comparison against alternative approaches.

---

## Acknowledgments

- **University of Aveiro** — academic supervision (Prof. Luís Seabra Lopes)
- **APS — Administração dos Portos de Sines e do Algarve, S.A.** — problem statement, real-world datasets, and collaboration under the NEXUS programme
- **Open-source community** — SentenceTransformers, FAISS, Pandas
