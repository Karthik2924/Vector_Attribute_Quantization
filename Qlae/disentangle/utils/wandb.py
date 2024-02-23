import omegaconf
import wandb
import os
import hydra

def initialize_wandb(config, name,dir = " "):
    if dir == " ":
        dir = '/home/ksanka/Research/latent_quantization-Copy1/latent_quantization/'
    run = wandb.init(
        project=config.wandb.project,
        config=omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        save_code=True,
        group=config.wandb.group,
        job_type=config.wandb.job_type,
        name=config.wandb.name if config.wandb.name is not None else name,
        dir = dir
    )
    wandb.config.update({'wandb_run_dir': wandb.run.dir})
    wandb.config.update({'hydra_run_dir': os.getcwd()})
    return run