import easydict


config_dict = easydict.EasyDict({
    "input_dim": 768,
    "num_patch": 1600,
    "model_dim": 768,
    "ffn_dim": 1024,
    "attention_heads": 6,
    "attention_dropout": 0.1,
    "dropout": 0.4,
    "encoder_layers": 4,
    "decoder_layers": 4,
})