import numpy as np

cm = np.array([
    [2, 0, 0],  
    [0, 1, 0],  
    [0, 0, 1]   
])

# Calculate accuracy
accuracy = np.trace(cm) / np.sum(cm)  

# Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR)

FAR_class0 = cm[0, 1:].sum() / cm[0].sum() if cm[0].sum() != 0 else 0
FAR_class1 = cm[1, [0, 2]].sum() / cm[1].sum() if cm[1].sum() != 0 else 0
FAR_class2 = cm[2, :2].sum() / cm[2].sum() if cm[2].sum() != 0 else 0

# Average FAR across all classes
FAR = (FAR_class0 + FAR_class1 + FAR_class2) / 3

# FRR for each class
FRR_class0 = cm[1:, 0].sum() / cm[:, 0].sum() if cm[:, 0].sum() != 0 else 0
FRR_class1 = cm[[0, 2], 1].sum() / cm[:, 1].sum() if cm[:, 1].sum() != 0 else 0
FRR_class2 = cm[:2, 2].sum() / cm[:, 2].sum() if cm[:, 2].sum() != 0 else 0

# Average FRR across all classes
FRR = (FRR_class0 + FRR_class1 + FRR_class2) / 3

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{cm}')
print(f'FAR: {FAR:.2f}')
print(f'FRR: {FRR:.2f}')
