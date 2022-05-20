from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch import Tensor, nn
from torchmetrics import PearsonCorrCoef, MeanSquaredError#, SpearmanCorrcoef
from scipy import stats
import sklearn.metrics as sm

def evaluate(model, test_loader, config, device):
    """
    ## Evaluate the trained model
    """

    loss, mse = test(model, test_loader, config, device)
    highest_losses, hardest_examples, true_labels, predictions, metrics = get_hardest_k_examples(model, test_loader.dataset, config, device)

    return loss, mse, highest_losses, hardest_examples, true_labels, predictions, metrics

def get_hardest_k_examples(model, testing_set, config, device, k=32):
    model.eval()

    loader = DataLoader(testing_set, 1, shuffle=False)

    # get the losses and predictions for each item in the dataset
    losses = None
    predictions = None
    gts = None
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader)):
              
            if "vit"==config["model_name"]:    
                im_tensor, batch_data = data  
                gt_score = batch_data["mean_score"]
                input = im_tensor.to(device)
            
            target = gt_score.to(device)
            output = model(input).squeeze(dim=1)
            loss = nn.MSELoss()(output, target)
            pred = output
            
            if losses is None:
                losses = loss.view((1, 1))
                predictions = pred
                gts = target
            else:
                losses = torch.cat((losses, loss.view((1, 1))), 0)
                predictions = torch.cat((predictions, pred), 0)
                gts = torch.cat((gts, target),0)

    argsort_loss = torch.argsort(losses, dim=0).squeeze()

    highest_k_losses = losses[argsort_loss[-k:]]
    hardest_k_examples = [testing_set[idx]['im_feature'] for idx in argsort_loss[-k:]]
    true_labels = [testing_set[idx]['im_score'] for idx in argsort_loss[-k:]]
    predicted_labels = predictions[argsort_loss[-k:]]

    metrics = get_metrics(labeller_score_list=gts.cpu(), network_score_list=predictions.cpu())

    return highest_k_losses, hardest_k_examples, true_labels, predicted_labels, metrics


def test(model, test_loader, config, device):
    model.eval()
    val_loss = 0
    val_sp, val_mse = 0, 0
    mean_squared_error = MeanSquaredError().to(device)

    count = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
                
            if "vit"==config["model_name"]:    
                im_tensor, batch_data = data  
                gt_score = batch_data["mean_score"]  
                input = im_tensor.to(device)
            
            gt_score = gt_score.to(device)
            pred_score = model(input).squeeze(dim=1)

            if config["loss"] == "mse":
                loss = nn.MSELoss()(pred_score, gt_score)

            elif config["loss"] == "spearman":
                raise NotImplementedError()

            val_loss += loss.detach()
            val_mse +=mean_squared_error(pred_score, gt_score).detach()
            count += 1

    avg_val_loss = val_loss / count
    avg_val_mse = val_mse / count

    return avg_val_loss, avg_val_mse

def get_metrics(labeller_score_list: Tensor, network_score_list: Tensor, verbose=True) -> dict:
    srcc = stats.spearmanr(labeller_score_list, network_score_list)
    print("SRCC =", srcc)
    mse = round(sm.mean_squared_error(labeller_score_list, network_score_list), 4)
    print("MSE =", mse)
    lcc = stats.pearsonr(labeller_score_list, network_score_list)
    print("LCC =", lcc)

    return {"SRCC": srcc,
            "MSE": mse,
            "LCC": lcc}