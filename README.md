# Global Life Expectancy Trends & Prediction
## Project Overview

This project delves into global life expectancy trends using the OECD Health Status dataset. Leveraging Python for data analysis and machine learning, and Power BI for interactive visualization, the aim is to uncover insights into life expectancy patterns influenced by country, year, and gender, and to build a predictive model for future trends.

## Problem Statement

Understanding and predicting life expectancy is crucial for public health planning, policy-making, and assessing overall societal well-being. This project addresses the challenge of extracting meaningful insights from complex health datasets to identify key drivers and trends in life expectancy across various demographics and geographical regions.

## Dataset

* **Name:** OECD Health Status - Life Expectancy at 0 years
* **Source:** OECD (Organisation for Economic Co-operation and Development)
* **File:** `OECD.ELS.HD,DSD_HEALTH_STAT@DF_HEALTH_STATUS,1.1+.A.LFEXP.Y.Y0.........csv`
* **Description:** This dataset provides life expectancy at birth (0 years) for various OECD countries, disaggregated by year and gender. It serves as a rich source for analyzing demographic health indicators.

## Methodology

The project follows a robust analytical pipeline, implemented primarily in Python:

### 1. Data Cleaning & Preprocessing (`src/health.py`)
-   **Handling Missing Values:** Irrelevant, empty, or constant columns were systematically identified and removed.
-   **Inconsistent Formats:** Columns were renamed for clarity and consistency (e.g., `REF_AREA` to `Country_Code`, `OBS_VALUE` to `Observation_Value`).
-   **Data Transformations:** Features were prepared for machine learning, including one-hot encoding for categorical variables (`Country_Name`, `Gender`, `Age_Group`) and standardization for numerical variables (`Year`).
-   **Output:** The cleaned dataset is saved as `data/OECD_Health_Status_Cleaned.csv`.

### 2. Exploratory Data Analysis (EDA) (`src/health.py`, `visuals/`)
-   Generated descriptive statistics for life expectancy.
-   Visualized the distribution of life expectancy across the dataset.
-   Analyzed life expectancy trends over time, segmented by gender.
-   Compared average life expectancy across different countries.
-   All generated plots are saved in the `visuals/` directory.

### 3. Machine Learning Model Application (`src/health.py`)
-   **Model Choice:** A `RandomForestRegressor` was chosen as a suitable model for predicting continuous life expectancy values, known for its robustness and ability to capture complex relationships.
-   **Training:** The model was trained on the preprocessed dataset, using `Country_Name`, `Age_Group`, `Gender`, and `Year` as features.

### 4. Model Evaluation (`src/health.py`)
-   The model's performance was evaluated using standard regression metrics:
    -   Mean Absolute Error (MAE)
    -   Mean Squared Error (MSE)
    -   Root Mean Squared Error (RMSE)
    -   R-squared (R2) Score
-   A sample of actual vs. predicted values is also provided to illustrate prediction accuracy.

### 5. Innovation (`src/health.py`, `visuals/`)
-   To enhance the project's insights, **Feature Importance** from the `RandomForestRegressor` model was extracted and visualized. This step helps identify which factors (e.g., Country, Year, Gender) contribute most significantly to the variations in life expectancy predictions, offering valuable insights beyond just prediction accuracy. The plot is saved in `visuals/feature_importance.png`.

## Power BI Dashboard (`powerbi/`)

An interactive Power BI dashboard has been developed to visualize the key insights derived from the data analysis.

### Dashboard Features:
-   **Problem & Insights:** Clearly communicates the overall trends, country-specific comparisons, and gender disparities in life expectancy.
-   **Interactivity:** Incorporates slicers for `Year`, `Country_Name`, and `Gender` to allow users to dynamically explore the data.
-   **Appropriate Visuals:** Utilizes line charts for trends, bar charts for comparisons, and KPI cards for key metrics.
-   **Design Clarity:** Adheres to a consistent color theme, clear labels, and a tidy layout for an intuitive user experience.
-   **Innovative Features:** (Describe any specific unique elements in your Power BI design, e.g., custom tooltips, specific drill-throughs, or advanced DAX measures if you implemented any)

### How to View the Dashboard:
1.  Ensure you have Power BI Desktop installed.
2.  Download the `YourDashboardName.pbix` file from the `powerbi/` directory in this repository.
3.  Open the `.pbix` file with Power BI Desktop.
4.  Navigate through the report pages and interact with the slicers and visuals.

## How to Run the Code (Reproduce the Analysis)

To replicate the data analysis and generate the output files:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-project-repo-name.git](https://github.com/your-username/your-project-repo-name.git)
    cd your-project-repo-name
    ```
2.  **Place the Dataset:**
    Download the raw dataset (`OECD.ELS.HD,DSD_HEALTH_STAT@DF_HEALTH_STATUS,1.1+.A.LFEXP.Y.Y0.........csv`) and place it inside the `data/` directory.
3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
4.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn tabulate
    ```
5.  **Run the Python Script:**
    ```bash
    python src/health.py
    ```
    This script will:
    -   Load and clean the data.
    -   Save the cleaned data to `data/OECD_Health_Status_Cleaned.csv`.
    -   Generate EDA plots in the `visuals/` directory.
    -   Train and evaluate the Machine Learning model.
    -   Generate a feature importance plot in the `visuals/` directory.

## Key Insights / Findings (Example - you will fill this with your actual findings)

* Global life expectancy has shown a general upward trend over the observed years.
* There's a consistent gender gap in life expectancy, with females typically living longer than males across most observed countries.
* Significant variations in average life expectancy exist among OECD countries, highlighting potential areas for policy intervention.
* (Add specific insights from your feature importance analysis, e.g., "Year was the most significant predictor of life expectancy," or "Country played a substantial role in accounting for variations.")

## Technologies Used

* **Python:** For data analysis, cleaning, EDA, and machine learning.
    * Pandas (Data manipulation)
    * NumPy (Numerical operations)
    * Scikit-learn (Machine learning)
    * Matplotlib (Plotting)
    * Seaborn (Statistical data visualization)
    * Tabulate (Markdown table formatting)
* **Power BI:** For interactive dashboard creation and visualization.

## Future Work / Enhancements

* Explore more advanced regression models or ensemble techniques.
* Integrate additional datasets (e.g., GDP, healthcare expenditure) to understand other factors influencing life expectancy.
* Develop a time-series forecasting model for life expectancy.
* Deploy the predictive model as a web application.

## Contact

[Your Name] - [Your Email/LinkedIn Profile Link]

---

**Remember to:**
* Replace `your-project-repo-name` and `your-username` with your actual GitHub details.
* Replace `YourDashboardName.pbix` with the actual name of your Power BI file.
* **Add actual screenshots** of your Power BI dashboard and key Python plots to the `visuals/` and `powerbi/dashboard_screenshots/` folders, and link them in the README.md using Markdown image syntax (`![Alt Text](path/to/image.png)`).
* **Fill in the "Key Insights / Findings" section** with the actual discoveries you make during your analysis.
