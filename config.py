from pathlib import Path

def get_config():
    return{
        #normal transformer
        "batch_size": 8,
        "h": 8,
        "N": 6,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",

        #extra variables for speculative decoding
        "q_d_model": 128,
        "q_h": 4,
        "q_N": 2,
        "gamma": 4, #how many new predictions
        "q_preload": None,
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)