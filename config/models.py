# Configuration of various language models

models = {
    "partial_brnn": {
        "module": "models.partial_brnn.model",
        # used to generate batches from the tensor
        "generator": {
            "function": "partial_brnn",
            "extra": {
                "extra_args": ["seq_length", "back_steps"],
                "data_loader": ["vocab"],
            }
        },
        "processor": "default_process",
        "eval_processor": {
            "function": "partial_brnn",
            "extra": {
                "extra_args": ["back_steps"]
            }
        },
        "summary": ""
    },
    "brnn_gap": {
        "module": "models.brnn_gap.model",
        # used to generate batches from the tensor
        "generator": {
            "function": "brnn_gap",
            "extra": {
                "extra_args": ["seq_length"],
                "data_loader": ["vocab"],
            }
        },
        "processor": "default_process",
        "eval_processor":{
            "function": "brnn_gap",
            "extra": {
                "extra_args": []
            }
        },
        "summary": ""
    }
}