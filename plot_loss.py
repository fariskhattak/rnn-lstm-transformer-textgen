import json
import matplotlib.pyplot as plt

model_type = "lstm"
loss_path = f"loss_data/{model_type}_loss.json"

with open(loss_path, "r") as f:
    loss_data = json.load(f)

train_losses = loss_data["train_losses"]
val_losses = loss_data["val_losses"]

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", marker='o')
plt.plot(val_losses, label="Val Loss", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{model_type.upper()} Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
