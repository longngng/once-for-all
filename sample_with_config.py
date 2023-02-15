import torch
from torch.utils.data import DataLoader
import hydra
import os
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix, classification_report

from ofa.imagenet_classification.elastic_nn.networks import OFAProxylessNASNets
from ofa.imagenet_classification.data_providers.mitbih import ECGDataset
from ofa.utils import count_net_flops, count_parameters
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics


def load_dataset(path, batch_size, shuffle, dims=2):
    dataset = ECGDataset(file_path=path, dims=dims)
    dataset_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    return dataset_loader


def eval_model(net, test_loader):
    """
    Eval a pytorch model on a test set.
    Returns accuracy and a classification report
    """
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


def load_pth_model(saved_pth_dict_path, val_loader, model_config):
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

    init = torch.load(saved_pth_dict_path, map_location="cpu")["state_dict"]
    ofa.load_state_dict(init)
    ofa.re_organize_middle_weights()

    ofa.set_active_subnet(
        [int(i) for i in model_config["ks"]],
        [int(i) for i in model_config["e"]],
        [int(i) for i in model_config["d"]],
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
    return model


def convert_pth_to_onnx(net, test_path, onnx_model_path):
    def get_sample_input(test_path, batch_size=1):
        test_loader = load_dataset(
            path=test_path,
            batch_size=batch_size,
            shuffle=True,
            dims=2,
        )
        for data in test_loader:
            inputs, _ = data
            return inputs.float()

    if not os.path.exists(os.path.dirname(onnx_model_path)):
        os.makedirs(os.path.dirname(onnx_model_path))

    torch.onnx.export(
        net.float(),  # model being run
        get_sample_input(test_path),  # model input (or a tuple for multiple inputs)
        onnx_model_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=13,  # the ONNX version to export the model to #11 or higher due to onnx2tf requirements
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )
    return onnx_model_path


@hydra.main(config_path="configs", config_name="sample_subnetwork.yaml")
def main(cfg: DictConfig):
    # Change batch_size below will only affect the result slightly, almost no difference
    val_loader = load_dataset(cfg.val_csv_path, batch_size=512, shuffle=True)

    net = load_pth_model(cfg.saved_pth_dict_path, val_loader, cfg.model_config)

    # # Change batch_size below does not affect the result at all
    # test_loader = load_dataset(cfg.test_csv_path, batch_size=512, shuffle=True)

    # eval_model(net, test_loader)
    convert_pth_to_onnx(net, cfg.test_csv_path, cfg.output_onnx_model)


if __name__ == "__main__":
    main()
