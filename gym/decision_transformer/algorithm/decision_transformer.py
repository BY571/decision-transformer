import torch
from torch import nn, Tensor

class BaseAlgo():
    def __init__(self,):
        pass

    def save_ckpt(self, ):
        raise NotImplementedError
    
    def load_ckpt(self, ):
        raise NotImplementedError

    def get_action(self, ):
        raise NotImplementedError

    def train(self, ):
        raise NotImplementedError

class DecisionTransformer(BaseAlgo):
    def __init__(self, predictor, config):
        super().__init__()
        self.predictor = predictor.to(config.device)
        warmup_steps = config.algorithm.warmup_steps
        self.optimizer = torch.optim.AdamW(self.predictor.parameters(),
                                      lr=config.algorithm.learning_rate,
                                      weight_decay=config.algorithm.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lambda steps: min((steps+1)/warmup_steps, 1))
        
        self.loss_fn = nn.MSELoss()
        
    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        self.set_eval_mode()
        with torch.no_grad():
            return self.predictor.get_action(states, actions, rewards, returns_to_go, timesteps, **kwargs)
    
    def set_train_mode(self, ):
        self.predictor.train()
        
    def set_eval_mode(self, ):
        self.predictor.eval()

    def train(self, states, actions, rewards, rtg, timesteps, attention_mask):

        action_target = torch.clone(actions)
        
        action_preds = self.predictor.forward(states,
                                              actions,
                                              rewards,
                                              rtg[:,:-1],
                                              timesteps,
                                              attention_mask=attention_mask)

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(action_preds, action_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), .25)
        self.optimizer.step()

        return loss.detach().cpu().item()