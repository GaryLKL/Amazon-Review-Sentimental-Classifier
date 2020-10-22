# Use the below script at line 2 to run this file
# PYTHONPATH="/scratch/kll482/cathay:$PYTHONPATH" python testing.py

import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import torch
from src.train_test.train_utils import test, model_load, write_log, write_config
from src.models.gcn_conv import BinGCNConv
from src.models.gat_conv import BinGATConv

print("1. Prepare for initial settings")
file_name = "edgeIndex1_gat_2020_09_12_14_11"
model_type = "graph"
config_saved_path = "result/{}/config_saved/{}.pkl".format(model_type, file_name)

# resume my config file
file = open(config_saved_path, 'rb')
config = pickle.load(file)

# resume the test loader
test_loader = torch.load(os.path.join(config.data_loader_path,
                                      "test_loader_{}_{}.pth".format(config.embedd_method,
                                                                     config.edge_index)))

# initialize testing model
if config.model_name == "gcn":
    model_init = BinGCNConv(config, config.num_features, config.n_classes).to(config.device)
elif config.model_name == "gat":
    model_init = BinGATConv(config, config.num_features, config.n_classes).to(config.device)
model_test = model_load(config=config, 
                        model_test=model_init,
                        name=file_name,
                       )

# start testing the model on the test loader
print("2. Start testing")
y_true, y_pred, y_prob = test(config, test_loader, model_test)
print("Finished testing")

# save the testing result

predicted_result_path = os.path.join(config.result_path, "prediction/{}.pkl".format(file_name))

with open(predicted_result_path, 'wb') as f:
    pickle.dump([y_true, y_pred, y_prob], f)

# write the classification report
print("3. Write the classification report...")
report = classification_report(y_true, y_pred)
write_log("\n{}\n\n{}\n".format("=== Classification Report ===",
                                report),
          config.log_file,
          "a+")

# reload the training and validation loss for showing the learning curve
with open(config.log_file, "r+") as file:
    logs = file.readlines()
    val_loss = []
    train_loss = []
    for row in logs:
        if len(row.split("|")) < 2:
            continue
        else:
            train_loss.append(float(row.split("|")[-2].strip().split(" ")[-1]))
            val_loss.append(float(row.split("|")[-1].strip().split(" ")[-1]))
    assert len(train_loss) == len(val_loss)

# draw the learning curve
print("4. Plot the learning curve and save it...")
fig, ax = plt.subplots()
ax.plot(range(len(train_loss)), train_loss, '-b', label='train')
ax.plot(range(len(val_loss)), val_loss, '--r', label='validation')
leg = ax.legend()
plt.title("Learning Rate")
plt.ylabel("Loss")
plt.xlabel("Per 500 Batch")

plot_path = os.path.join(config.result_path, "learning_curve/{}.png".format(file_name))
fig.savefig(plot_path)
