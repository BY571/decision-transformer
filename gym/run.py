import hydra
from omegaconf import OmegaConf
import wandb
from decision_transformer.trainer import Trainer

@hydra.main(config_path='config', config_name="conf")
def main(cfg):
    trainer = Trainer(cfg)
    print(OmegaConf.to_yaml(cfg))
    
    with wandb.init(project=cfg.project, name=cfg.run_name, mode=cfg.mode, config=cfg):
        trainer.train(wandb)
        
    
if __name__ == "__main__":
    main()