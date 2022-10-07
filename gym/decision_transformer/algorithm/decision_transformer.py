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
        self.scheduler.step()

        return {"training/loss": loss.detach().cpu().item()}

class OnlineDecisionTransformer(BaseAlgo):
    def __init__(self, predictor, config):
        super().__init__()
        
        self.predictor = predictor.to(config.device)
        self.target_entropy = - self.predictor.act_dim
        warmup_steps = config.algorithm.warmup_steps
        self.optimizer = torch.optim.AdamW(self.predictor.parameters(),
                                      lr=config.algorithm.learning_rate,
                                      weight_decay=config.algorithm.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lambda steps: min((steps+1)/warmup_steps, 1))
        self.log_entropy_multiplier = torch.zeros(1, requires_grad=True, device=config.device)
        self.multiplier_optimizer = torch.optim.AdamW(
            [self.log_entropy_multiplier],
            lr=config.algorithm.learning_rate,
            weight_decay=config.algorithm.weight_decay,
        )

        self.multiplier_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.multiplier_optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1))
            
        
        self.loss_fn = lambda a_log_prob, entropies: -torch.mean(a_log_prob) - torch.exp(self.log_entropy_multiplier.detach()) * torch.mean(entropies)
        self.entropy_loss_fn = lambda entropies: torch.exp(self.log_entropy_multiplier) * (torch.mean(entropies.detach()) - self.target_entropy)

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        self.set_eval_mode()
        with torch.no_grad():
            return self.predictor.get_action(states, actions, rewards, returns_to_go, timesteps, **kwargs)
    
    def set_train_mode(self, ):
        self.predictor.train()
        
    def set_eval_mode(self, ):
        self.predictor.eval()

    def train(self, states, actions, rewards, rtg, timesteps, attention_mask):

        # action_target = torch.clone(actions)

        _, action_log_probs, entropies = self.predictor.forward(states,
                                              actions,
                                              rewards,
                                              rtg[:,:-1],
                                              timesteps,
                                              attention_mask=attention_mask)

        # act_dim = action_preds.shape[2]
        # action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        # action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_log_probs = action_log_probs.reshape(-1)[attention_mask.reshape(-1) > 0]
        entropies = entropies.reshape(-1)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(action_log_probs, entropies)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), .25)
        self.optimizer.step()
        
        entropy_loss = self.entropy_loss_fn(entropies)
        self.multiplier_optimizer.zero_grad()
        entropy_loss.backward()
        self.multiplier_optimizer.step()
        
        return {"training/loss": loss.detach().cpu().item(),
                "training/entropy_loss": entropy_loss.detach().cpu().item(),
                "training/entropy_multiplier": torch.exp(self.log_entropy_multiplier).detach().cpu().item(),
                "training/entropy": torch.mean(entropies).item()}