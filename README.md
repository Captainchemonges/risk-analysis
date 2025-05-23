Thank you for sharing the sample rows and details from your `aviation_data.csv`. To make the README more **concrete and actionable**, Ill tailor it specifically to your dataset and project context, emphasizing the exact columns, data characteristics, and practical steps.

---

# README.md - Aviation Accident Risk Analysis and Aircraft Selection

---

## Project Overview

This project analyzes the **NTSB aviation accident dataset** to identify aircraft manufacturers with the lowest relative risk, supporting a company entering the aviation industry in selecting safer aircraft for purchase and operation.

The dataset spans multiple years and includes detailed records of aviation accidents and incidents, with key attributes such as aircraft make/model, injury counts, weather conditions, and flight phases.

---

## Dataset Description

- **File:** `aviation_data.csv`
- **Sample Columns Used:**
  - `Event_Id`: Unique accident/incident identifier
  - `Event_Date`: Date of accident (YYYY-MM-DD)
  - `Make`: Aircraft manufacturer (e.g., CESSNA, BOEING)
  - `Model`: Aircraft model
  - `Total_Fatal_Injuries`: Number of fatal injuries in the accident
  - `Weather_Condition`: Weather at time of accident (e.g., VMC, IMC, UNKNOWN)
  - `Phase_of_Flight`: Flight phase during accident (e.g., TAKEOFF, LANDING, CRUISE)
  - `Injury_Severity`: Severity category (Non-Fatal, Fatal, etc.)

*Note:* The dataset contains some missing or inconsistent values (e.g., blank weather or phase), which are handled during cleaning.

---

## Data Cleaning & Preparation

- **Standardized `Make` values** by converting to uppercase and trimming whitespace to unify naming variants (e.g., `Boeing` and `BOEING` treated as the same).
- **Converted `Event_Date` to datetime** for accurate temporal analysis.
- **Filled missing values** in `Weather_Condition` and `Phase_of_Flight` with `"UNKNOWN"`.
- **Converted injury counts to numeric**, replacing missing values with zeros to avoid calculation errors.

---

## Risk Scoring Model

To quantify manufacturer risk, the model aggregates accident data by `Make` and computes:

- **Total Accidents:** Count of accidents per manufacturer.
- **Total Fatalities:** Sum of fatal injuries per manufacturer.
- **Average Fatalities per Accident:** Severity metric = Total Fatalities / Total Accidents.

These metrics are **normalized** using Min-Max scaling to a 01 range, then combined into a **weighted risk score**:

| Metric                    | Weight |
|---------------------------|--------|
| Total Accidents           | 40%    |
| Total Fatalities          | 40%    |
| Average Fatalities/Accident | 20%    |

Manufacturers are ranked by risk score descending; a higher score indicates greater relative risk.

---

## Visualizations

- **Top Aircraft Makes by Risk Score:**
  Horizontal bar chart showing the top 10 manufacturers with the highest risk scores, highlighting those with the greatest safety concerns.

- **Accident Frequency by Weather Condition and Flight Phase:**
  Grouped bar chart showing accident counts by weather (e.g., VMC = Visual Meteorological Conditions) and phase of flight (e.g., TAKEOFF, LANDING), illustrating high-risk operational contexts.

---

## Business Recommendations

1. **Acquire Aircraft from Manufacturers with Lower Risk Scores:**
   Prioritize purchasing aircraft from manufacturers with consistently lower risk scores to reduce operational risk and improve safety.

2. **Strengthen Pilot Training and Safety Protocols for High-Risk Conditions:**
   Focus on flight phases and weather conditions with elevated accident frequencies (e.g., TAKEOFF and LANDING under VMC), implementing targeted training and operational procedures.

3. **Implement Continuous Risk Monitoring:**
   Regularly update risk scores and accident trends with new data to inform fleet management and safety decisions proactively.

---

## How to Run the Analysis

1. **Install dependencies:**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn streamlit
   ```

2. **Run the Jupyter Notebook:**

   - Load and clean data
   - Calculate risk scores
   - Generate visualizations
   - Review business recommendations

3. **Run the Interactive Dashboard:**

   ```bash
   streamlit run aviation_dashboard.py
   ```

   Use sidebar filters to explore risk scores and accident patterns interactively.

---

## File Structure

```
/project-root

   aviation_data.csv               # Raw accident dataset
   aviation_data_cleaned.csv       # Cleaned dataset (optional)
   aviation_analysis_notebook.ipynb # Analysis notebook
   aviation_dashboard.py           # Streamlit dashboard app
   README.md                      # This documentation
```

---

## Notes on Dataset Specifics

- **Weather Conditions:** Mostly reported as `VMC` (Visual Meteorological Conditions), `IMC` (Instrument Meteorological Conditions), or `UNKNOWN`.
- **Phases of Flight:** Includes phases such as `TAKEOFF`, `LANDING`, `CRUISE`, `MANEUVERING`, and others.
- **Injury Severity:** Includes categories like `Non-Fatal`, `Fatal`, and `Destroyed` aircraft.

---

## References
NTSB Aviation Accident Database

Aviation Safety Reporting System (ASRS)

Industry best practices for aviation safety and risk management

---

## Contact

For questions or collaboration:

- **Name:** Peter Chemonges
- **Email:** chemongeskip7@gmail.com
- **Organization:** Moringa Institute

---



---


---
