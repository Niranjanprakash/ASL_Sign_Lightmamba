import json
from pathlib import Path
from backend.config import OUTPUT_DIR

def save_ablation_results(experiment_name: str, config_switches: dict, test_metrics: dict):
    """
    Saves the ablation experiment results to outputs/experiments/.
    """
    exp_dir = OUTPUT_DIR / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_data = {
        "experiment_name": experiment_name,
        "config_switches": config_switches,
        "metrics": test_metrics
    }
    
    file_path = exp_dir / f"{experiment_name}.json"
    with open(file_path, "w") as f:
        json.dump(experiment_data, f, indent=4)
        
    print(f"[ABLATION] Saved experiment results to {file_path}")
