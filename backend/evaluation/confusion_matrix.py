import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import json
from backend.config import OUTPUT_DIR, CLASSES

def generate_confusion_analysis(targets: np.ndarray, predictions: np.ndarray):
    """
    Generates a 10-class confusion matrix, saves the plot, 
    and identifies frequently confused sign pairs.
    """
    cm_dir = OUTPUT_DIR / "confusion_matrix"
    cm_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(targets, predictions, labels=list(range(len(CLASSES))))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=CLASSES, yticklabels=CLASSES,
           title="LightMamba-ASL Confusion Matrix",
           ylabel="True label",
           xlabel="Predicted label")
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    fig.tight_layout()
    plt.savefig(cm_dir / "confusion_matrix.png")
    plt.close()
    
    # Analyze confused pairs (where i != j and cm[i, j] > 0)
    confusions = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "class_a": CLASSES[i],
                    "class_b": CLASSES[j],
                    "a_predicted_as_b": int(cm[i, j]),
                    "b_predicted_as_a": int(cm[j, i]),
                    "combined_score": int(cm[i, j] + cm[j, i])
                })
                
    # Sort by combined score descending
    confusions = sorted(confusions, key=lambda x: x["combined_score"], reverse=True)
    
    # Save confusion report
    report_path = cm_dir / "confusion_report.json"
    with open(report_path, "w") as f:
        json.dump(confusions, f, indent=4)
        
    print("\n" + "="*50)
    print("TOP CONFUSED CLASS PAIRS")
    print("="*50)
    for item in confusions[:5]:
        print(f"{item['class_a']} <-> {item['class_b']} | Combined Confusions: {item['combined_score']} (A->B: {item['a_predicted_as_b']}, B->A: {item['b_predicted_as_a']})")
    print("="*50)
