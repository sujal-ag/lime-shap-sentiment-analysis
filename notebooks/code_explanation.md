# LIME vs SHAP Comparative Analysis — Code Explanation

## 🧩 Step-by-Step Breakdown

### 1. Install and Import Libraries
- Installs lime and shap.
- Imports necessary libraries: pandas, numpy, sklearn, lime, shap, matplotlib, etc.
- Suppresses warnings for clean output.

---

### 2. Upload and Load Dataset
- Prompts user to upload IMDB Dataset.csv.
- Loads dataset into a pandas DataFrame.
- Converts text labels (positive/negative) into numeric (1/0).

---

### 3. Train-Test Split
- Splits the data into 80% training and 20% testing.
- Resets indexes for consistent access.

---

### 4. Text Preprocessing and Model Setup
- Uses *TF-IDF Vectorizer* to transform text into numerical features (max 500 features).
- Uses *Logistic Regression* as the classifier.
- Combines both into a pipeline for easy training and prediction.

---

### 5. Train and Evaluate Model
- Fits the pipeline on training data.
- Calculates and prints accuracy on test data (~84%).

---

### 6. Initialize Explainers
- Creates LimeTextExplainer for LIME.
- SHAP explainer is created later dynamically per sample.

---

### 7. Run Explanations (20 Random Samples)
- Randomly selects 20 test samples.
- For each sample:
  - Gets prediction and confidence.
  - Runs *LIME* (10 features, 300 perturbations).
  - Runs *SHAP* (30 background samples).
  - Extracts top 10 features from each.
  - Measures explanation time.
- Stores results in a DataFrame.

---

### 8. Compute Comparison Metrics
- Calculates:
  - *Jaccard Similarity* = overlap between top features.
  - *Average Feature Overlap* (out of 10).
  - *Average explanation times* for LIME & SHAP.
  - *Speed ratio* (SHAP slower by ~3.9×).

---

### 9. Create Visualizations
- Generates a 2×2 figure:
  1. Jaccard distribution histogram.
  2. Time comparison boxplot.
  3. LIME top features bar chart.
  4. SHAP top features bar chart.
- Saves the figure as lime_vs_shap_analysis.png.

---

### 10. Generate Summary Report
- Prints and saves a text summary (analysis_summary.txt) including:
  - Model accuracy
  - Feature agreement
  - Timing results
  - Recommendations and limitations

---

### 11. Display Example Explanation
- Prints:
  - Sample review text
  - True vs predicted label
  - Top 5 LIME & SHAP features
  - Number of overlapping features

---

### 12. Final Checklist
- Prints completion message.
- Lists generated files:
  - lime_vs_shap_analysis.png
  - analysis_summary.txt
- Displays final key metrics (samples, agreement, speed, accuracy).

---

## ✅ In Short
This code:
1. Trains a *Logistic Regression + TF-IDF* model on IMDB reviews.  
2. Generates *LIME* and *SHAP* explanations for 20 samples.  
3. Compares them using *Jaccard similarity, **timing, and **feature overlap*.  
4. Produces a visual comparison and summary report.

*Key Output Files:*
- 📊 lime_vs_shap_analysis.png
- 📄 analysis_summary.txt