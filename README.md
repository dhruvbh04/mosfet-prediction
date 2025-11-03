# Predicting MOSFET Threshold Voltage using Machine Learning

This repository contains the project files for "Predicting MOSFET Threshold Voltage using Machine Learning," prepared in partial fulfillment of the requirements for the course EEE G513 at BITS Pilani, Goa Campus.

The project implements a machine learning framework to automatically and objectively predict the threshold voltage ($V_{th}$) of MOSFET devices from their $I_{D}-V_{G}$ characteristics. This data-driven approach addresses the inconsistencies and subjective nature of traditional extraction methods like the Constant Current (CC) or Second Derivative methods.

The methodology is based on the research paper: **“Automatic Prediction of MOSFETs Threshold Voltage by Machine Learning Algorithms” (EDTM 2023)**.

---

## Dataset

The project uses the **MESD (MOSFET Electrical Simulation Dataset)**, an open-source dataset containing JSON files of simulated $I_{D}-V_{G}$ curves for over 40 device types.

* **Source:** [SJTU-YONGFU-RESEARCH-GRP/MESD-MOSFET-Electrical-Simulation-Dataset](https://github.com/SJTU-YONGFU-RESEARCH-GRP/MESD-MOSFET-Electrical-Simulation-Dataset)

The `mosfet_pred_26oct.ipynb` notebook clones this repository directly to load the data.

---

## Methodology

The end-to-end workflow is implemented in the Jupyter notebook and follows these key steps:

1.  **Data Loading:** Loads all `*.json` simulation files from the MESD dataset. One faulty file (`N180A-pmos1.json`) is skipped.
2.  **Label Extraction:** The "ground truth" $V_{th}$ for each curve is extracted using the **Constant Current (CC) method**, which finds the $V_{GS}$ at a predefined current ($I_{D} = 10^{-7} A$).
3.  **Target Normalization:** To make the model device-agnostic, the extracted $V_{th}$ is converted into a normalized **Threshold Ratio ($R_{th}$)** between 0 and 1. This $R_{th}$ serves as the target variable (label) for the ML models.
    $$R_{th} = \frac{V_{th} - V_{G,min}}{V_{G,max} - V_{G,min}}$$
4.  **Feature Engineering (Dual Normalization):** The $I_{D}-V_{G}$ input curves are processed into a fixed-length feature vector:
    * The $V_{GS}$ axis is linearly normalized to a [0, 1] range.
    * The $log_{10}(|I_{D}|)$ is taken.
    * The curve is **resampled to a 50-point vector** via linear interpolation.
    * This 50-point vector is min-max scaled to [0, 1] to create the final feature vector.
5.  **Model Training:** The processed dataset (136,423 valid curves) is split into 70% training and 30% testing sets. Two models are trained:
    * **k-Nearest Neighbors (kNN) Regressor** ($n\_neighbors=5$)
    * **Decision Tree Regressor** ($max\_depth=10$)
6.  **Reconstruction:** A function is provided to predict $R_{th}$ from a feature vector and then convert it back to the physical $V_{th}$ using the device's $V_{GS}$ range.

---

## Results

The kNN model demonstrated superior performance and generalization compared to the Decision Tree.

| Model | RMSE | MAE | R² (Coefficient of Determination) |
| :--- | :--- | :--- | :--- |
| **k-Nearest Neighbors** | **0.0613** | **0.0344** | **0.9295** |
| Decision Tree | 0.0752 | 0.0492 | 0.8938 |
*(Metrics based on predicting the normalized Rth on the 30% test set).*



---

## How to Run

1.  **Environment:** Ensure you have Python and the required libraries installed:
    ```bash
    pip install numpy pandas scikit-learn matplotlib
    ```
2.  **Notebook:** Open and run the `mosfet_pred_26oct.ipynb` notebook (e.g., in Jupyter or Google Colab).
3.  **Data:** The notebook will automatically clone the MESD dataset from GitHub when you run the data loading cell.
4.  **Execution:** Run all cells sequentially to load data, preprocess features, train the models, and visualize the results.

---

## Authors & Acknowledgements

**Authors:**
* Ayan Sinha (2022B5A81024G)
* Dhruv Bhardwaj (2022B4A31271G)
* Mannat Koolwal (2022B5AA0085G)

**Guide:**
* Prof. Shivin Srivastava

This project was submitted in partial fulfillment of the requirements for the course EEE G513 at BITS Pilani, Goa Campus.
