import torch



class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, config.num_layers * config.hidden_size * 2)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_layers * config.hidden_size * 2)
    
    def forward(self, device, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix).to(device)
            # prefix_tokens.to(device)
            past_key_values = self.trans(prefix_tokens).to(device)
        else:
            past_key_values = self.embedding(prefix).to(device)
        return past_key_values


def get_prompt_prefix_tuning(batch_size, device, config, dtype=torch.half):

    prefix_encoder = PrefixEncoder(config, device)
    prefix_tokens = torch.arange(config.pre_seq_len).long()
    prefix_tokens = prefix_tokens.to(device)
    dropout = torch.nn.Dropout(0.1)

    


    prefix_tokens = prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
    past_key_values = prefix_encoder(device, prefix_tokens).type(dtype)
    past_key_values = past_key_values.to(device)
    past_key_values = past_key_values.view(
        batch_size,
        config.pre_seq_len,
        config.num_layers * 2,
        config.num_attention_heads,
        config.hidden_size // config.num_attention_heads
    )
    # seq_len, b, nh, hidden_size
    past_key_values = dropout(past_key_values)
    past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
    # past_key_values = [(v[0], v[1]) for v in past_key_values]
    return past_key_values