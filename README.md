# SMBJ Classifier: Single Molecule Break Junction DNA Classification

## Overview

This repository contains the implementation of a machine learning-based classification system for Single Molecule Break Junction (SMBJ) DNA analysis. Developed during my research internship at the University of Washington, this project demonstrates the application of advanced machine learning techniques to classify DNA strands based on their electrical conductance signatures.

## Research Background

### Single Molecule Break Junction (SMBJ) Technology

Single Molecule Break Junction is a cutting-edge technique that exploits the conductive properties of DNA molecules. When DNA strands are stretched between two electrodes, they exhibit unique electrical conductance patterns that can be measured as current-voltage relationships. These conductance signatures are characteristic of specific DNA sequences and structures, making them ideal for molecular identification and classification.

### The Challenge

While DNA strands exhibit distinct conductance patterns, these differences are often imperceptible to the human eye when visualized as conductance graphs. Traditional analysis methods struggle to distinguish between different DNA variants, especially when dealing with subtle sequence variations or structural differences.

### The Solution: AI-Powered Classification

This project addresses this challenge by implementing multiple machine learning approaches to automatically classify DNA strands based on their conductance signatures. The system can distinguish between different DNA variants with remarkable accuracy, achieving classification accuracies exceeding 99.9% in optimal configurations.

## Research Context

This work was conducted during my research internship at the University of Washington, focusing on the application of machine learning to molecular electronics and DNA characterization. The research demonstrates the potential of AI-driven approaches in molecular biology and nanotechnology applications.

## Dataset

The system was trained and tested on COVID-19 Alpha variant DNA strands, specifically designed to distinguish between:

- **Alpha_MM1**: Alpha variant with mismatch pattern 1
- **Alpha_MM2**: Alpha variant with mismatch pattern 2  
- **Alpha_PM**: Alpha variant with perfect match

Each dataset contains conductance traces collected under various experimental conditions including different voltage biases (50mV-200mV), current amplifier settings (1nAV-10nAV), and ramp rates (3-20).

## Methodology

### Data Preprocessing Pipeline

1. **Current Trace Processing**: Raw current traces are filtered using low-pass filters (LPF) optimized for 10kHz and 30kHz sampling frequencies
2. **Quality Control**: R-squared filtering removes traces with poor exponential decay fits (R² < 0.95)
3. **Conductance Conversion**: Current traces are converted to conductance values using the quantum conductance unit (G₀ = 7.748091729 × 10⁻⁵ S)
4. **Histogram Generation**: Conductance traces are converted into 1D or 2D histograms for machine learning input

### Machine Learning Approaches

The system implements 8 different classification approaches:

#### Approach A1: 1D Histograms + XGBoost
- **Accuracy**: 79.93% - 90.51% (depending on sample size)
- Uses individual conductance histograms with XGBoost classifier

#### Approach A2: 1D Averaged Histograms + XGBoost ⭐
- **Accuracy**: 79.93% - 99.49% (depending on sample size)
- Uses averaged conductance histograms across datasets within the same variant
- **Best Performance**: 99.49% accuracy with 50 samples

#### Approach A3: 2D Histograms + CNN+XGBoost
- Combines 2D conductance-distance histograms with Convolutional Neural Networks
- CNN extracts features, XGBoost performs final classification

#### Approach A4: 2D Averaged Histograms + CNN+XGBoost
- Uses averaged 2D histograms for improved generalization

#### Approach A5: 1D Histograms + Random Forest 🌟
- **Accuracy**: 95.31% - 99.96% (depending on sample size)
- **Best Performance**: 99.96% accuracy with 150 samples
- Excellent performance with ensemble learning

#### Approach A6: K-Means Clustering
- **Accuracy**: 34.26% - 45.60%
- Unsupervised clustering approach for exploratory analysis

#### Approach A7: 1D Histograms + CatBoost 🏆
- **Accuracy**: 97.91% - 99.97% (depending on sample size)
- **Best Performance**: 99.97% accuracy with 150 samples
- **Highest Overall Performance**

#### Approach A8: 1D Histograms + LightGBM
- **Accuracy**: 99.81% - 99.94% (depending on sample size)
- **Best Performance**: 99.94% accuracy with 150 samples
- Excellent gradient boosting performance

## Key Results

### Performance Summary

| Approach | Model | Best Accuracy | Sample Size |
|----------|-------|---------------|-------------|
| A7 | CatBoost | **99.97%** | 150 |
| A5 | Random Forest | **99.96%** | 150 |
| A8 | LightGBM | **99.94%** | 150 |
| A2 | XGBoost (Averaged) | **99.49%** | 50 |

### Sample Size Impact

The research demonstrates that classification accuracy improves significantly with larger sample sizes:

- **5 samples**: 79.93% accuracy
- **10 samples**: 90.51% accuracy  
- **25 samples**: 95.31% - 97.91% accuracy
- **50 samples**: 98.73% - 99.59% accuracy
- **75 samples**: 99.53% - 99.83% accuracy
- **100 samples**: 99.81% - 99.92% accuracy
- **150 samples**: 99.94% - 99.97% accuracy

## Technical Implementation

### Dependencies

- **Deep Learning**: TensorFlow 2.15+, Keras 3.0+
- **Machine Learning**: scikit-learn 1.4+, XGBoost 2.0+, CatBoost, LightGBM
- **Scientific Computing**: NumPy 1.26+, SciPy 1.12+
- **Visualization**: Matplotlib 3.8+

### Architecture

```
SMBJClassifier/
├── SMBJClassifier/
│   ├── DPP.py          # Data Preprocessing Pipeline
│   ├── model.py        # Machine Learning Models
│   └── __init__.py
├── demo/
│   ├── COVID_Alpha_4_approaches.py  # Main execution script
│   ├── Data/           # COVID Alpha dataset
│   └── Result/         # Classification results
└── tests/
```

### Usage

```python
from SMBJClassifier import DPP, model

# Load and preprocess data
data, Amp, Freq, RR, Vbias = DPP.readInfo('COVID_Strand_Source.txt', 'COVID_Exp_parameter.mat')
DPP.createCondTrace(data, Amp, Freq, Vbias)

# Run classification
conf_mat = model.runClassifier(
    approach=7,  # CatBoost approach
    data=data,
    RR=RR,
    group=[[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]],
    num_group=3,
    sampleNum=150
)
```

## Research Impact

This work demonstrates several important contributions:

1. **High-Accuracy Classification**: Achieving 99.9%+ accuracy in DNA variant classification
2. **Multiple ML Approaches**: Comprehensive comparison of 8 different machine learning methods
3. **Sample Size Optimization**: Systematic analysis of how sample size affects classification performance
4. **Real-World Application**: Practical application to COVID-19 variant detection

## Future Directions

- Extension to other DNA variants and sequences
- Real-time classification capabilities
- Integration with experimental SMBJ setups
- Development of user-friendly interfaces for researchers

## Citation

If you use this code in your research, please cite:

```
SMBJ Classifier: Single Molecule Break Junction DNA Classification
Research conducted at University of Washington
[Valen Li], Research Intern
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this research or collaboration opportunities, please contact [li.valen.008@gmail.com].

---

*This research was conducted during my internship at the University of Washington, demonstrating the power of machine learning in molecular electronics and DNA characterization.*