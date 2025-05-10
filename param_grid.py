import itertools
import pandas as pd
from train import main as train_main
from argparse import Namespace

# 超参数搜索空间
param_grid = {
    "epochs": [5, 10],
    "batch_size": [32, 64],
    "lr_fc": [1e-3, 5e-4],
    "lr_backbone": [1e-4, 1e-5]
}

# models to be tuned
model_list = ["resnet18", "alexnet"]
init_type = "pretrained" # or "random"
data_dir = "caltech101_split"

def run_experiment(model_type, epochs, batch_size, lr_fc, lr_backbone):
    args = Namespace(
        data_dir=data_dir,
        model_type=model_type,
        init_type=init_type,
        batch_size=batch_size,
        epochs=epochs,
        lr_fc=lr_fc,
        lr_backbone=lr_backbone,
        tune_mode=True
    )
    acc = train_main(args)
    return acc

if __name__ == "__main__":
    combinations = list(itertools.product(
        param_grid["epochs"],
        param_grid["batch_size"],
        param_grid["lr_fc"],
        param_grid["lr_backbone"]
    ))

    for model_type in model_list:
        results = []
        print(f"\n==== Tuning model: {model_type} ====")

        for epochs, batch_size, lr_fc, lr_backbone in combinations:
            print(f"Trying: epochs={epochs}, batch_size={batch_size}, lr_fc={lr_fc}, lr_backbone={lr_backbone}")
            acc = run_experiment(model_type, epochs, batch_size, lr_fc, lr_backbone)
            results.append({
                "model": model_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr_fc": lr_fc,
                "lr_backbone": lr_backbone,
                "val_acc": acc
            })

        # save results to CSV
        df = pd.DataFrame(results)
        df.to_csv(f"tuning_results_{model_type}_{init_type}.csv", index=False)

        # print the best result for this model
        best_row = df.loc[df["val_acc"].idxmax()]
        print(f"\n>>> Best for {model_type}:")
        print(best_row)
