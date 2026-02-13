# Draw.io Diagrams for Stacking Ensemble ML Presentation

This directory contains professional Draw.io diagrams created for LaTeX presentations about the 2-level stacking ensemble methodology.

## 📊 Diagram Files

### 1. **01_stacking_architecture.drawio**
**Purpose:** Complete 2-level stacking architecture
**Content:**
- Input data flow
- K-Fold cross-validation process
- Level 0: 5 base models (Random Forest, SVM, Logistic, KNN, Naive Bayes)
- OOF (Out-Of-Fold) predictions generation
- Level 1: Meta-models (Ridge Regression, XGBoost)
- Final ensemble predictions
- Color-coded legend

**Best for:** Explaining the technical architecture and data flow

---

### 2. **02_results_comparison.drawio**
**Purpose:** Side-by-side comparison of results from 3 datasets
**Content:**
- Dataset 1: Ames Housing (2930 obs, 40 features)
  - Best Individual: RandomForest 94.36%
  - Best Stacking: XGBoost 94.53%
  - Gain: +0.17pp
  - Correlation: 0.944 (HIGH)

- Dataset 2: Pima Diabetes (768 obs, 8 features)
  - Best Individual: Logistic 76.47%
  - Best Stacking: XGBoost 73.20%
  - Gain: -3.27pp ⚠️
  - Correlation: 0.877 (HIGH)

- Dataset 3: Bank Marketing (41188 obs, 52 features)
  - Best Individual: Logistic 90.21%
  - Best Stacking: XGBoost 90.34%
  - Gain: +0.13pp
  - Correlation: 0.528 (LOW - better diversity)

**Best for:** Presenting experimental results and comparison

---

### 3. **03_methodology_workflow.drawio**
**Purpose:** Complete methodology from data loading to evaluation
**Content:**
- Phase 1: Data loading and preparation (3 datasets)
- Phase 2: Train/test split (80/20, stratified)
- Phase 3: Standardization (center + scale)
- Phase 4: Level 0 training with OOF (K=5 folds)
- Phase 5: Level 1 meta-model training
- Phase 6: Evaluation and comparison

**Best for:** Explaining the complete experimental methodology

---

### 4. **04_key_findings.drawio**
**Purpose:** Summary of key insights and conclusions
**Content:**
- Finding 1: Correlation is key (diversity matters)
- Finding 2: Dataset size is important (avoid overfitting)
- Finding 3: High dimensionality creates natural diversity
- Finding 4: OOF vs Blending comparison
- Finding 5: Practical recommendations
- General conclusion box

**Best for:** Conclusions and discussion section

---

### 5. **05_correlation_vs_performance.drawio**
**Purpose:** Scatter plot showing correlation vs stacking gain
**Content:**
- X-axis: Average correlation of Level 0 predictions
- Y-axis: Stacking gain in accuracy (pp)
- 3 data points (Bank, Pima, Ames)
- Negative trend line
- Favorable zone (low correlation, positive gain)
- Risky zone (high correlation, negative/low gain)

**Best for:** Visualizing the main discovery about correlation

---

## 🎨 How to Use

### Opening the Diagrams
1. Go to [draw.io](https://app.diagrams.net/) or use the desktop app
2. Click "File" → "Open"
3. Select the `.drawio` file you want to edit

### Exporting for LaTeX
1. Open the diagram in Draw.io
2. Click "File" → "Export as" → Choose format:
   - **PDF** (vector, best for LaTeX): Recommended
   - **PNG** (raster, high resolution): Set DPI to 300+
   - **SVG** (vector, scalable): Also good for LaTeX
3. Include in your LaTeX document:

```latex
% For PDF
\begin{figure}[h]
  \centering
  \includegraphics[width=0.9\textwidth]{diagrams/01_stacking_architecture.pdf}
  \caption{Architecture du stacking à 2 niveaux}
  \label{fig:stacking-arch}
\end{figure}

% For PNG
\begin{figure}[h]
  \centering
  \includegraphics[width=0.9\textwidth]{diagrams/02_results_comparison.png}
  \caption{Comparaison des résultats sur trois datasets}
  \label{fig:results-comp}
\end{figure}
```

### Customizing
All diagrams use:
- **Font:** Arial/Helvetica (standard)
- **Color scheme:**
  - Blue (#dae8fc): Data/Input
  - Green (#d5e8d4): Level 0 models
  - Purple (#e1d5e7): Level 1 meta-models
  - Red (#f8cecc): Output/Results
  - Yellow (#fff2cc): Process/Transform
  - Gray (#f5f5f5): Containers

To customize:
1. Open in Draw.io
2. Select elements
3. Change colors, text, or layout
4. Re-export

---

## 📈 Key Results Summary

| Dataset | Size | Features | Best Individual | Best Stacking | Gain | Correlation |
|---------|------|----------|----------------|---------------|------|-------------|
| Ames Housing | 2,930 | 40 | RF: 94.36% | XGB: 94.53% | +0.17pp | 0.944 (HIGH) |
| Pima Diabetes | 768 | 8 | LR: 76.47% | XGB: 73.20% | -3.27pp | 0.877 (HIGH) |
| Bank Marketing | 41,188 | 52 | LR: 90.21% | XGB: 90.34% | +0.13pp | 0.528 (LOW) |

**Main Conclusion:** Stacking works best when base model predictions have **low correlation** (< 0.7). High correlation means models are too similar, reducing stacking effectiveness.

---

## 💡 Presentation Tips

1. **Introduction:** Start with diagram 1 (Architecture) to explain the concept
2. **Methodology:** Use diagram 3 (Workflow) to describe your process
3. **Results:** Show diagram 2 (Comparison) for experimental results
4. **Analysis:** Use diagram 5 (Correlation vs Performance) to explain the key finding
5. **Conclusions:** End with diagram 4 (Key Findings) summarizing insights

---

## 📝 Citation

If you use these diagrams, please cite your research:

```bibtex
@misc{stacking_ensemble_2026,
  title={Stacking à 2 Niveaux vs Modèles Individuels: Analyse Comparative Multi-Datasets},
  author={Your Name},
  year={2026},
  note={Comparative study of 2-level stacking ensemble on Ames Housing, Pima Diabetes, and Bank Marketing datasets}
}
```

---

## 🔗 Related Files

- **Output CSV files:** `../output/*.csv`
- **Visualizations:** `../output/*.png`
- **Notebook:** `../notebooks/stacking_dual_dataset.ipynb`

---

## 📧 Contact

For questions or modifications, contact: [your.email@example.com]

---

**Last Updated:** February 13, 2026
**Version:** 1.0
**Format:** Draw.io XML (compatible with draw.io/diagrams.net)
