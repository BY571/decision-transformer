import transformers
from decision_transformer.models.trajectory_gpt2 import GPT2Model

def get_transformer(name: str="gpt2", embedding_dim: int=256, **kwargs):
    if name == "gpt2":
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=embedding_dim,
            **kwargs
        )
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        return GPT2Model(config)
    else:
        raise NotImplementedError