import easydict


config_dict = easydict.EasyDict({
    "image_size": 320,
    "conv_patch": 40,
    "model_dim": 896,
    "ffn_dim": 1024,
    "attention_heads": 4,
    "attention_dropout": 0.0,
    "dropout": 0.0,
    "encoder_layers": 0,
    "decoder_layers": 4,
})