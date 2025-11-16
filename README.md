[README.md](https://github.com/user-attachments/files/23569751/README.md)
# DIVOP: Variable-Order Path Classification Model

This repository contains the official code for the **DIVOP** model introduced in the paper:

**Secchini, Valeria, Javier Garcia-Bernardo, and Petr Janský (2025).** *"Avoiding Overfitting in Variable-Order Markov Models: a Cross-Validation Approach."* arXiv:2501.14476.

DIVOP is a model for detecting informative vs. non-informative paths using variable-order Markov representations, with emphasis on avoiding overfitting through cross-validation. The repository provides all necessary code and datasets (synthetic and real-flight data) to reproduce the experiments.

---

## Repository Structure

### **Data/**

This folder will contain all datasets used for training, validating, and testing the DIVOP model. However, training data are too large to be be uploaded on a GitHub repository, so you can download them from a Drive folder liked below.

1. **Synthetic datasets (training and testing)**

   * Available for both **Flight** and **Orbis** inspired data:

     * **Positives**: Sinthetic Informative paths 
             - Flight synthetic positives: "flight_positives_training_data.csv"
             - Orbis synthetic positives: "orbis_positives_training_data.csv"
     * **Negatives**: Synthetic Non-informative paths
             - Flight synthetic negatives: "flight_negatives_training_data.csv"
             - Orbis synthetic negatives: "orbis_negatives_training_data.csv"
      * **File size and download location**

     The synthetic training datasets are too large to be stored directly in this GitHub repository.  
     They are therefore hosted on Google Drive and must be downloaded manually.

     **Google Drive folder (all datasets):**  
     https://drive.google.com/drive/folders/1ql4W01b2NhC9E4QnRT8FQm0szBBhcL5c?usp=drive_link

     You can also download each file individually:

     - **Flight – Negative paths (synthetic)**  
       Actual file: `flight_negatives_training_data.csv.xz`  
       Direct download:  
       https://drive.google.com/uc?export=download&id=1QpW-m8cf8itSQ5uPf8Lw7eqJVJFAjegE  

     - **Flight – Positive paths (synthetic)**  
       Actual file: `flight_positives_training_data.csv`  
       Direct download:  
       https://drive.google.com/uc?export=download&id=1bPJd9AHwNQZK-cTUWEUxEOWjUrYWZFew  

     - **Orbis – Negative paths (synthetic)**  
       Actual file: `orbis_negatives_training_data.csv.xz`  
       Direct download:  
       https://drive.google.com/uc?export=download&id=18novmnPpu2c9O4zzV0gaSuKBjYbPuFUv  

     - **Orbis – Positive paths (synthetic)**  
       Actual file: `orbis_positives_training_data.csv`  
       Direct download:  
       https://drive.google.com/uc?export=download&id=13QydayWU0KtsHLvRP9I_TuIjqe4AxOLh  

     After downloading, place these files in the `Data/` folder to reproduce the experiments and run the notebooks.

2. **Real Flight Data**

   * Sourced from the HYPA repository (LaRock et al. 2020): "coupons_2018_01-5percent.ngram"
   * Provided to allow real-world evaluation.

3. **Real Orbis Data**

   * *Not included due to copyright restrictions.*
   * Instead, we include **synthetic Orbis-like datasets**, as mentioned before, which preserve the statistical structure of Orbis but not its internal proprietary information.



---

## Notebooks/

This folder contains the Python library and Jupyter notebooks used throughout the workflow to give example of usage.

### **1. `divop.py`**

A Python library containing all core functions used for:

* Synthetic data generation
* Classification of real and synthetic paths
* Construction and evaluation of DIVOP

### **2. `1_create_training_data.ipynb`**

Notebook designed to generate hundreds of synthetic datasets efficiently using the functions in `divop.py`. These synthetic datasets are required to train DIVOP.

### **3. `2_testing_data.ipynb`**

Notebook for evaluating the DIVOP model on synthetic datasets inspired by real data. This helps assess model performance before applying it to actual datasets.

### **4. `3_classifying_real_data.ipynb`**

Notebook for applying DIVOP to real datasets, such as the provided real flight data.

---

## Data Sources and Related Models

### **Flight Data (Real)**

Real flight data included in this repository is taken from the HYPA project:

* **LaRock, Timothy, et al. (2020).** *"Hypa: Efficient detection of path anomalies in time series data on networks."* SIAM SDM 2020.
* Original repository: [https://github.com/tlarock/hypa](https://github.com/tlarock/hypa)

### **Higher-Order Network Modeling Inspiration**

The `divop.py` library is inspired by the BuildHON model hon.py library:

* **Xu, Jian, Thanuka L. Wickramarathne, and Nitesh V. Chawla (2016).** *"Representing higher-order dependencies in networks."* Science Advances 2.5: e1600028.

---

## How to Use This Repository

1. Generate synthetic training datasets using `1_create_training_data.ipynb`.
2. Evaluate the model using `2_testing_data.ipynb`.
3. Apply DIVOP to real datasets using `3_classifying_real_data.ipynb`.


---

## Citation

If you use DIVOP in your research, please cite:

```
Secchini, Valeria, Javier Garcia-Bernardo, and Petr Janský.
"Avoiding Overfitting in Variable-Order Markov Models: a Cross-Validation Approach."
arXiv preprint arXiv:2501.14476 (2025).
```

---

For questions or clarifications, feel free to open an issue or get in touch to the email valeria.secchini.r@gmail.com
