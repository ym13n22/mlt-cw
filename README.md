This project implements a K-Means clustering model for unsupervised data classification, optimized using dimensionality reduction (PCA) and SSE evaluation with the Elbow Method to determine the optimal number of clusters.

🚀 Features

✅ Unsupervised learning with K-Means

📉 Dimensionality reduction using Principal Component Analysis (PCA)

🧠 Optimal cluster number selection via Elbow Method

📈 Model performance evaluated by Sum of Squared Errors (SSE)

⚡ Achieved ~85% improvement in computational efficiency after PCA

🧠 Methodology

Data Preprocessing

Normalized the dataset

Removed missing or noisy data

PCA (Principal Component Analysis)

Reduced feature dimensionality

Removed redundant information

Accelerated computation by ~85%

K-Means Clustering

Applied clustering on reduced data

Computed SSE for different K values

Elbow Method

Plotted SSE vs. K

Identified the "elbow point" as the optimal number of clusters

📊 Evaluation Metrics

SSE (Sum of Squared Errors):
Used to evaluate cluster compactness

Execution Time:
Compared before and after PCA for performance benchmarking

🧰 Tech Stack

Python (NumPy, Pandas, Scikit-learn, Matplotlib)
