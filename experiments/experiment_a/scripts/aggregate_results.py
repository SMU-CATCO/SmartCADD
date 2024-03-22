import bocas
import numpy as np
import pandas as pd

results = bocas.Result.load_collection("artifacts/")

all_dfs = []
for result in results:
    config = result.config
    history = result.get("history").metrics
    metrics = result.get("metrics").metrics

    cols = []
    col_heads = []

    col_heads += ["Size"]
    cols += [config.size]

    col_heads += ["Batch Size", "Max Epochs"]
    cols += [config.batch_size, config.epochs]

    col_heads += ["Used Epochs"]
    cols += [len(history["loss"])]

    col_heads += ["Test Accuracy"]
    test_acc = metrics["acc"]
    cols += [test_acc]

    col_heads += ["Validation Accuracy"]
    val_acc = np.array(history["val_acc"])
    best_epoch = np.argmax(val_acc)
    cols += [val_acc.max()]

    col_heads += ["Train Accuracy"]
    acc = np.array(history["acc"])[best_epoch]
    cols += [acc]

    col_heads += ["Version"]
    cols += [config.version]

    cols = [cols]
    all_dfs.append(pd.DataFrame(cols, columns=col_heads))

df = pd.concat(all_dfs)

# If you want to reorder
final_col_order = [
    "Size",
    "Test Accuracy",
    "Train Accuracy",
    "Validation Accuracy",
    "Batch Size",
    "Max Epochs",
    "Used Epochs",
    "Version",
]
df = df[final_col_order]
# Put best performers on the top and worst at the bottom
df = df.sort_values(by=["Test Accuracy"], ascending=False)
result = df.to_markdown()

with open("results/results.md", "w") as f:
    f.write("# Results\n")
    f.write(result)
