import ast
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from ofa.imagenet_classification.elastic_nn.networks import OFAProxylessNASNets
from ofa.imagenet_classification.data_providers.mitbih import ECGDataset
from ofa.utils import count_net_flops, count_parameters
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics


def eval(net, test_loader):
    net.eval()
    correct = 0
    total = 0

    y_pred = []
    y_true = []

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            output = torch.argmax(outputs, dim=1).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
    target_names = ["class 0", "class 1", "class 2", "class 3", "class 4"]
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
    print("Accuracy of the network on the test data: %.3f" % (correct / total))
    return (correct / total), report


def load_dataset(path, batch_size, shuffle, dims=2):
    dataset = ECGDataset(file_path=path, dims=dims)
    dataset_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    return dataset_loader


def get_bin_id(total_macs):
    MIN_MACS = 180000
    MAX_MACS = 940000
    if total_macs < MIN_MACS:
        return 0
    elif total_macs > MAX_MACS:
        return 11
    else:
        return (int)((10) * (total_macs - MIN_MACS) / (MAX_MACS - MIN_MACS))


ofa = OFAProxylessNASNets(
    n_classes=5,
    bn_param=(0.1, 1e-5),
    dropout_rate=0.1,
    base_stage_width="proxyless",
    width_mult=0.4,
    ks_list=[3, 5],
    expand_ratio_list=[2, 3, 4],
    depth_list=[1, 2, 3],
)

model_path = "exp/kernel_depth2kernel_depth_width/phase2/checkpoint/checkpoint.pth.tar"
init = torch.load(model_path, map_location="cpu")["state_dict"]
ofa.load_state_dict(init)
ofa.re_organize_middle_weights()

# Change batch_size below will only affect the result slightly, almost no difference
val_loader = load_dataset("dataset/271022/mitbih_val.csv", batch_size=512, shuffle=True)

macs_list = []
param_list = []
acc_list = []
report_acc_list = []
report_macro_precision_list = []
report_macro_recall_list = []
report_macro_f1_list = []
report_wa_precision_list = []
report_wa_recall_list = []
report_wa_f1_list = []

model_config_list = []

model = ofa.get_active_subnet()
model.double()
set_running_statistics(model, val_loader)

TOTAL_MACS_SUPERNETWORK = count_net_flops(model.float(), data_shape=[1, 1, 1, 260])
print("TOTAL_MACS_SUPERNETWORK", TOTAL_MACS_SUPERNETWORK)
model.double()
acc, report = eval(model, val_loader)

count = 0
NUM_OF_SAMPLES = 100
macs_histogram = {}

df = pd.read_csv("configs_050123.csv", index_col=0)

for index, row in df.iterrows():
    model_config = ast.literal_eval(row["configs"])

    print(model_config)
    ofa.set_active_subnet(
        model_config["ks"],
        model_config["e"],
        model_config["d"],
    )

    model = ofa.get_active_subnet()
    model.double()
    set_running_statistics(model, val_loader)
    total_macs = count_net_flops(model.float(), data_shape=[1, 1, 1, 260])

    model.double()

    print(model_config)
    total_params = count_parameters(model)
    print("total_macs", total_macs)
    print("total_params", total_params)
    acc, report = eval(model, val_loader)

    report_macro_precision_list.append(report["macro avg"]["precision"])
    report_macro_recall_list.append(report["macro avg"]["recall"])
    report_macro_f1_list.append(report["macro avg"]["f1-score"])
    report_wa_precision_list.append(report["weighted avg"]["precision"])
    report_wa_recall_list.append(report["weighted avg"]["recall"])
    report_wa_f1_list.append(report["weighted avg"]["f1-score"])
    report_acc_list.append(report["accuracy"])

    macs_list.append(total_macs)
    param_list.append(total_params)
    acc_list.append(acc)
    model_config_list.append(model_config)
    count += 1

    df_data_dict = {
        "total_macs": macs_list,
        "total_params": param_list,
        "acc": acc_list,
        "report_acc": report_acc_list,
        "macro_precision": report_macro_precision_list,
        "macro_recall": report_macro_recall_list,
        "macro_f1": report_macro_f1_list,
        "wa_precision": report_wa_precision_list,
        "wa_recall": report_wa_recall_list,
        "wa_f1": report_wa_f1_list,
        "configs": model_config_list,
    }
    pd.DataFrame.from_dict(df_data_dict).to_csv("acc_versus_size_ofa_proxyless.csv")
