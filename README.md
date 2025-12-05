# Walmart Weekly Sales Forecasting (Time Series Analysis)
**Authors:** Daniel Woodward & Olivia Pisano

### Project Overview
This project analyzes and forecasts weekly sales for Walmart stores to identify the most accurate modeling techniques for retail demand planning. By applying a "Tournament" style approach using the Nixtla suite of libraries, this analysis aims to determine which class of models—Statistical, Machine Learning, Deep Learning, or Foundation Models—performs best on high-volatility retail data.

The core of the project involves preprocessing sales data with exogenous variables (CPI, fuel price, temperature), training multiple forecasting models (including AutoNBEATS, LightGBM, and TimeGPT), and evaluating them using rigorous cross-validation metrics like RMSE and MAPE.

### Files in This Repository
- **ISA_444_Project.ipynb**: The main Jupyter Notebook containing all Python code for data loading, preprocessing, model training, and evaluation.
- **train.csv / test.csv**: The primary datasets containing historical weekly sales figures.
- **features.csv / stores.csv**: Supplementary datasets providing exogenous variables (holidays, economic indicators) and store metadata.
- **final_evaluation_output.csv**: The generated output file containing the detailed cross-validation predictions for all models.
- **final_metrics_summary.csv**: A summary table ranking models by global accuracy metrics (RMSE, MAE, MAPE).
- **testing_outputs.csv**: The final future forecasts generated for the testing horizon.

### Methodology
The analysis was conducted using Python within a Jupyter Notebook, utilizing the Nixtla ecosystem. The key steps are as follows:

1. **Data Loading & Preprocessing**: Sales data was merged with economic indicators. Missing values were interpolated, and the dataset was downsampled to the top 20 high-volume series for computational efficiency.
2. **Model Training (The Tournament)**: We implemented a diverse set of models:
    * **Statistical**: Naive, SeasonalNaive, AutoETS, AutoARIMA.
    * **Machine Learning**: LightGBM (Gradient Boosting) with lag features.
    * **Deep Learning**: AutoNBEATS and AutoNHITS (Neural Hierarchical Interpolation).
    * **Foundation Models**: TimeGPT (Generative AI for Time Series).
3. **Cross-Validation**: Models were evaluated using a rolling window approach (5 windows, 4-week horizon) to simulate real-world forecasting scenarios.
4. **Evaluation**: We compared models based on Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE) to determine the global winner and per-series winners.

### Results
The analysis identified **AutoNBEATS** (Deep Learning) as the superior model for this dataset, achieving the lowest global RMSE (~9,782) and MAPE (~5.17%). While Deep Learning won the global metrics, simple baselines like SeasonalNaive still performed best on specific individual series, highlighting the importance of model diversity.

### How to Run This Project
To replicate this analysis, you must use a **Linux-based environment**.

**⚠️ Important Environment Note:**
This project utilizes "Auto" models (AutoNBEATS/AutoNHITS) from the `neuralforecast` library, which rely on Ray for multiprocessing. **This code cannot be run on VS Code (Windows) or a local Windows machine.** It must be executed in a Linux environment, such as **Google Colab**.

**🔐 Security Note regarding API Keys:**
You will notice a hardcoded API key for TimeGPT in the notebook. This key is part of a temporary, school-based subscription managed by our professor for the ISA 444 course. It is included strictly for academic grading purposes and will be invalid shortly after the project concludes.

1. **Open in Google Colab**: Upload the `ISA_444_Project.ipynb` file to Google Colab.

2. **Upload Data**: Ensure `train.csv`, `test.csv`, `features.csv`, and `stores.csv` are uploaded to the Colab session storage.

3. **Install Dependencies**: Run the first cell to install the required libraries:
   - `%pip install pandas numpy matplotlib seaborn statsforecast mlforecast neuralforecast lightgbm nixtla`

4. **Run the Notebook**: Execute the cells in order. The script will preprocess the data, run the training pipeline, and generate the evaluation CSV files in the runtime directory.
