import gym
from torch import nn, Tensor

from decision_transformer.predictors.utils import get_transformer
from decision_transformer.predictors.decision_transformer import DTPredictor, StochDTPredictor
from decision_transformer.algorithm.decision_transformer import DecisionTransformer, OnlineDecisionTransformer, OnlineDecisionTransformerRND, RNDDecisionTransformer
from buffer.trajectory_buffer import ReplayBuffer

def get_env(env_name):
    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
    else:
        raise NotImplementedError
    return env, env.observation_space.shape[0], env.action_space.shape[0]

def get_predictor(state_dim, act_dim, config):
    transformer = get_transformer(name=config.algorithm.predictor.name,
                                    embedding_dim=config.algorithm.predictor.hidden_size,
                                    n_layer=config.algorithm.predictor.n_layer,
                                    n_head=config.algorithm.predictor.n_head,
                                    n_inner= config.algorithm.predictor.n_inner,
                                    activation_function=config.algorithm.predictor.activation_function,
                                    n_positions=config.algorithm.predictor.n_positions,
                                    resid_pdrop=config.algorithm.predictor.resid_pdrop,
                                    attn_pdrop= config.algorithm.predictor.attn_pdrop)
    if config.algorithm.predictor.name == "gpt2":
        return DTPredictor(state_dim=state_dim,
                           act_dim=act_dim,
                           transformer=transformer,
                           hidden_size=config.algorithm.predictor.hidden_size,
                           remove_pos_embs=config.algorithm.predictor.remove_pos_embs,
                           max_length=config.eval_context,
                           max_ep_len=config.env.max_ep_len,
                           action_tanh=config.algorithm.predictor.action_tanh)
    elif config.algorithm.predictor.name == "stochastic_gpt2":
        return StochDTPredictor(state_dim,
                                act_dim=act_dim,
                                transformer=transformer,
                                hidden_size=config.algorithm.predictor.hidden_size,
                                max_length=config.eval_context,
                                max_ep_len=config.env.max_ep_len,
                                action_tanh=config.algorithm.predictor.action_tanh,
                                log_std_min=config.algorithm.predictor.log_std_min,
                                log_std_max=config.algorithm.predictor.log_std_max,
                                remove_pos_embs=config.algorithm.predictor.remove_pos_embs,
                                stochastic_tanh=config.algorithm.predictor.stochastic_tanh,
                                approximate_entropy_samples=config.algorithm.predictor.approximate_entropy_samples)
    else: 
        raise NotImplementedError


def get_algorithm(state_dim, act_dim, config):
    if config.algorithm.name == "dt":
        return DecisionTransformer(get_predictor(state_dim, act_dim, config), config)
    elif config.algorithm.name == "online-dt":
        return OnlineDecisionTransformer(predictor=get_predictor(state_dim, act_dim, config), config=config)
    elif config.algorithm.name == "rnd-odt":
        return OnlineDecisionTransformerRND(predictor=get_predictor(state_dim, act_dim, config), config=config)
    elif config.algorithm.name == "rnd-dt": 
        return RNDDecisionTransformer(predictor=get_predictor(state_dim, act_dim, config), config=config)
    else:
        raise NotImplementedError

def get_buffer(state_dim, act_dim, config):
    if config.buffer.name == "rtgbuffer":
        return ReplayBuffer(max_size=config.buffer.max_size,
                          state_dim=state_dim,
                          action_dim=act_dim,
                          batch_size=config.batch_size,
                          max_len=config.algorithm.predictor.K,
                          max_ep_len=config.env.max_ep_len,
                          device=config.device,
                          rtg_scale=config.env.scale,
                          state_mean=config.env.state_mean,
                          state_std=config.env.state_std)
    else:
        raise NotImplementedError