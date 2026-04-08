Bank Customer Churn Prediction with PyTorch

A binary classification project that predicts whether a bank customer will leave (churn) using a custom feedforward Neural Network built from scratch with PyTorch.

Dataset
The project uses the **Churn Modelling** dataset containing 10,000 bank customer records with 14 features including credit score, geography, gender, age, balance, and more. The target variable is `Exited` (1 = churned, 0 = stayed).

What I Did
- Explored the dataset using `info()`, `describe()`, and `head()`
- Dropped irrelevant identifier columns (`RowNumber`, `CustomerId`, `Surname`)
- Applied **One-Hot Encoding** to categorical features (`Gender`, `Geography`)
- Split data into train/test sets (80/20)
- Standardized features using **StandardScaler**
- Converted data to PyTorch tensors with GPU support (CUDA if available)
- Built and trained a custom **Neural Network** with:
  - Input layer (13 features) → Linear(16) → Dropout(0.2) → Linear(8) → Linear(1)
  - ReLU activations + Sigmoid output
  - **BCELoss** + **Adam** optimizer (lr=0.001)
  - 100 training epochs

Results
| Set | Accuracy |
|---|---|
| Training | 79.46% |
| Test | **80.40%** |

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (preprocessing)
- PyTorch

## How to Run
1. Clone the repo
2. Install dependencies: `pip install pandas numpy scikit-learn torch`
3. Add the `Churn_Modelling_.csv` dataset to the project folder
4. Run the notebook