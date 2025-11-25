# Predicting MOSFET Threshold Voltage using Machine Learning

This repository contains the project files for "Predicting MOSFET Threshold Voltage using Machine Learning," prepared in partial fulfillment of the requirements for the course EEE G513 at BITS Pilani, Goa Campus.

The project implements a machine learning framework to automatically and objectively predict the threshold voltage ($V_{th}$) of MOSFET devices from their $I_{D}-V_{G}$ characteristics. This data-driven approach addresses the inconsistencies and subjective nature of traditional extraction methods like the Constant Current (CC) or Second Derivative methods.

The methodology is based on the research paper: **“Automatic Prediction of MOSFETs Threshold Voltage by Machine Learning Algorithms” (EDTM 2023)**.

---

## Dataset

The project uses the **MESD (MOSFET Electrical Simulation Dataset)**, an open-source dataset containing JSON files of simulated $I_{D}-V_{G}$ curves for over 40 device types.

* **Source:** [SJTU-YONGFU-RESEARCH-GRP/MESD-MOSFET-Electrical-Simulation-Dataset](https://github.com/SJTU-YONGFU-RESEARCH-GRP/MESD-MOSFET-Electrical-Simulation-Dataset)

The `mosfet_prediction.ipynb` notebook clones this repository directly to load the data.

---

## Methodology

The end-to-end workflow is implemented in the Jupyter notebook and follows these key steps:

1.  **Data Loading & Filtering:** * Loads `*.json` simulation files from the MESD dataset.
    * Filters data to include only devices with Process Corner = `tt` (Typical-Typical).
    * Removes curves that are flat lines or invalid.
2.  **Label Extraction:** * The "ground truth" $V_{th}$ for each curve is extracted using the **2nd Derivative Method**, which finds the peak of the second derivative of $log(I_D)$ with respect to $V_{GS}$.
3.  **Target Normalization:** * To make the model agnostic to specific voltage sweeps, the extracted $V_{th}$ is converted into a normalized **Threshold Ratio ($R_{th}$)** between 0 and 1.
    $$R_{th} = \frac{V_{th} - V_{G,min}}{V_{G,max} - V_{G,min}}$$
4.  **Feature Engineering:** * The $I_{D}-V_{G}$ input curves are processed into a fixed-length feature vector of **50 points**.
    * $V_{GS}$ is normalized to [0, 1].
    * Current is transformed to $log_{10}(|I_{D}|)$ and interpolated to a fixed axis.
    * The resulting vector is min-max scaled.
5.  **Device-Specific Training:** * The dataset is split by **Device Type** (NMOS and PMOS).
    * Separate models are trained for each device type to maximize accuracy.
    * Algorithms used: **k-Nearest Neighbors (kNN)** and **Decision Tree Regressor**.
6.  **Prediction & Denormalization:** * The system predicts the ratio $R_{th}$ and mathematically converts it back to the physical voltage $V_{th}$ using the input curve's voltage range.

---

## Results

The dataset was split into 70% training and 30% testing. The **k-Nearest Neighbors (kNN)** model consistently outperformed the Decision Tree across both device types.

### Performance Metrics (Test Set)

| Device Type | Model | RMSE | MAE | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- |
| **NMOS** | **kNN** | **0.0214** | **0.0039** | **0.9669** |
| NMOS | Decision Tree | 0.0364 | 0.0136 | 0.9040 |
| **PMOS** | **kNN** | **0.0207** | **0.0043** | **0.9675** |
| PMOS | Decision Tree | 0.0339 | 0.0133 | 0.9126 |

### Validation
A random sample validation of 100 curves was performed using the trained kNN models.
* **Accuracy:** 95% of predictions were within a tolerance of **0.001 V (1 mV)** of the calculated ground truth.

---

## How to Run

1.  **Environment:** Ensure you have Python and the required libraries installed:
    ```bash
    pip install numpy pandas scikit-learn matplotlib
    ```
2.  **Notebook:** Open and run the `mosfet_prediction_universal.ipynb` notebook (e.g., in Jupyter or Google Colab).
3.  **Data:** The notebook handles data acquisition automatically by cloning the MESD repository.
4.  **Execution:** Run all cells sequentially. The notebook will:
    * Preprocess the data.
    * Train separate models for NMOS and PMOS.
    * Output accuracy metrics.
    * Visualize predicted vs. actual values.
    * Run a random sampling validation test.

---

## Authors & Acknowledgements

**Authors:**
* Ayan Sinha (2022B5A81024G)
* Dhruv Bhardwaj (2022B4A31271G)
* Mannat Koolwal (2022B5AA0085G)

**Guide:**
* Prof. Shivin Srivastava

This project was submitted in partial fulfillment of the requirements for the course EEE G513 at BITS Pilani, Goa Campus.