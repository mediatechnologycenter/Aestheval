from tqdm import tqdm 
import torch.optim as optim
from torchmetrics import PearsonCorrCoef, MeanSquaredError
from torch import nn
import torch
from aestheval.baselines.training.build_models import save_model, load_model

def train(model, train_loader, valid_loader, config, device):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
    
    best_val_loss = None
    if config["load_existing_model"]:
        best_val_loss = load_model(model, optimizer=optimizer, config=config, device=device)

    mean_squared_error = MeanSquaredError().to(device)
    example_ct = 0
    for epoch in range(config["continue_from_epoch"], config["num_epochs"]):
        model.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for batch_idx, data in enumerate(tepoch):
                
                tepoch.set_description(f"Epoch {epoch}")
                
                if "vit" == config["model_name"]:
                    im_tensor, gt_score = data  
                    input = im_tensor.to(device)
                
                gt_score = gt_score.to(device)
                pred_score = model(input).squeeze(dim=1) # don't squeeze batch axis

                if config["loss"] == "mse":
                    loss = nn.MSELoss()(pred_score, gt_score)

                elif config["loss"] == "spearman":
                    raise NotImplementedError()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                mse = mean_squared_error(pred_score, gt_score)

                tepoch.set_postfix(loss=loss.item(), mse = mse.item())

                example_ct += len(data)

                if batch_idx % config["batch_log_interval"] == 0 and batch_idx > 0:
                    # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #     epoch, batch_idx * len(data), len(train_loader.dataset),
                    #     100. * batch_idx / len(train_loader), loss.item()))
                    
                    train_log(loss, example_ct, epoch)

        if epoch % config["save_model_every"] == 0 and epoch>0:
            save_model(model, epoch, optimizer, loss, best_val_loss, config, best_model=False)
        
        # evaluate the model on the validation set at each epoch
        val_loss, val_mse, best_val_loss = val(model, valid_loader, config, device, optimizer, best_val_loss, epoch)  
        test_log(val_loss, val_mse, example_ct, epoch)

def val(model, test_loader, config, device, optimizer,best_val_loss, current_epoch):
    model.eval()
    val_loss = 0
    val_sp, val_mse = 0, 0
    mean_squared_error = MeanSquaredError().to(device)

    count = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
               
            if "vit"==config["model_name"]:    
                im_tensor, gt_score = data  
                input = im_tensor.to(device)
            
            gt_score = gt_score.to(device)
            pred_score = model(input).squeeze(dim=1)

            if config["loss"] == "mse":
                loss = nn.MSELoss()(pred_score, gt_score)

            elif config["loss"] == "spearman":
                raise NotImplementedError()

            val_loss += loss.detach()
            # if args.batch_size >= 1:
            #     val_sp += spearman(pred_score, gt_score).detach()
            # else: val_sp = 0
            val_mse +=mean_squared_error(pred_score, gt_score).detach()
            count += 1

    avg_val_loss = val_loss / count
    avg_val_mse = val_mse / count

    if best_val_loss is None or avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_model(model, current_epoch, optimizer, best_val_loss, best_val_loss, config, best_model=True)

    
    return avg_val_loss, avg_val_mse, best_val_loss



def train_log(loss, example_ct, epoch):
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    

def test_log(loss, mse, example_ct, epoch):
    loss = float(loss)
    mse = float(mse)
    print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{mse:.3f}")