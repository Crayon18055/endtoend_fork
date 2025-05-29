import easydict


config_dict = easydict.EasyDict({
    "input_dim": 1792,
    "num_patch": 1600,
    "model_dim": 1792,
    "ffn_dim": 2048,
    "attention_heads": 4,
    "attention_dropout": 0.0,
    "dropout": 0.0,
    "encoder_layers": 0,
    "decoder_layers": 4,
})