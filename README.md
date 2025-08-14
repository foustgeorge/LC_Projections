# DataJoint-Based Pipeline for Sleep Recording and Fiber Photometry Analysis in Rodents

This repository provides Python code to build and run a **DataJoint-powered data pipeline** for analyzing **polysomnographic recordings (EEG and EMG)** alongside **fiber photometry measurements**. The example focuses on recordings from the **locus coeruleus (LC)** in the mouse brain, enabling structured ingestion, management, and analysis of combined sleep and photometry datasets.

## Features

- **Automated data ingestion** from CSV metadata and MATLAB `.mat` files
- **Support for multiple experimental paradigms:**
  - Baseline fiber photometry
  - Sleep deprivation (SD)
  - Sensory-enhanced sleep deprivation (SSD)
  - Fear conditioning (training, recall, extinction)
- **Integration with DataJoint** for relational database management
- **Robust handling of missing values** and standardized metadata formats

## Requirements

- Python 3.x
- [DataJoint](https://github.com/datajoint/datajoint-python). On how to run Datajoint locally using Docker please follow the provided documentation at the website
- pandas, numpy, tqdm, pathlib, glob
- mat73 (for reading MATLAB `.mat` files)
