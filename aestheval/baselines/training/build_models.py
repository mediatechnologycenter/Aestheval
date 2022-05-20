from multiprocessing.sharedctypes import Value
import timm
from torch import nn
from pathlib import Path
import torch

def _build_model(config):

    assert isinstance(config, dict)

    if config["model_name"] == "vit":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        model.reset_classifier(num_classes=1)

    if config['verbose']:    
        print("Params to learn:")
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    return model

def save_model(network, epoch, optimizer, loss, best_val_loss, config, best_model=False):
    if epoch:
        print("Saving model epoch: ", epoch)
    if best_model:
        print(f"Saving new best model at epoch {epoch}")
        
    if not Path(config["this_run_checkpoints"]).exists():
        Path(config["this_run_checkpoints"]).mkdir(exist_ok=True)

    if best_model:
        save_dir = Path(config["this_run_checkpoints"], "best_model.pt")
    else:
        save_dir = Path(config["this_run_checkpoints"], f"model_{epoch}_epoch.pt")
    torch.save(
        {
            "epoch": epoch,
            "model": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_loss": loss,
            "best_val_loss": best_val_loss,
        },
        save_dir,
    )


def load_best_model(model, this_run_checkpoints):
    chkpt_path = Path(this_run_checkpoints, "best_model.pt")
    checkpoint = torch.load(chkpt_path)
    print(f"Best model in epoch {checkpoint['epoch']}")
    model.load_state_dict(checkpoint["model"])

def load_model(network, optimizer, config, device):
    
    assert config.best_model or config.load_epoch or config.chkpt_path

    if not Path(config.this_run_checkpoints).exists() or not any(Path(config.this_run_checkpoints).iterdir()):
        raise ValueError("This checkpoint path does not exist or has no weights: ", config.this_run_checkpoints)

    if config.best_model:
        assert isinstance(config.best_model,  bool)
        chkpt_path = Path(config.this_run_checkpoints, "best_model.pt")

    elif config.load_epoch:
        assert isinstance (config.load_epoch, int)
        chkpt_path = Path(config.this_run_checkpoints, f"model_{config.load_epoch}_epoch.pt")
    
    elif config.chkpt_path:
        assert isinstance (config.load_epoch, str)
        chkpt_path = config.chkpt_path

    checkpoint = torch.load(chkpt_path, map_location=device)
    network.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    config.update({'continue_from_epoch': checkpoint["epoch"] + 1}, allow_val_change=True)
    
    print(
        "-> loaded checkpoint %s (epoch: %d)"
        % (chkpt_path, config.continue_from_epoch - 1)
    )
    best_val_loss = checkpoint["best_val_loss"]
    print(
        f"-> Previous best val loss: {best_val_loss}"
    )
    return best_val_loss