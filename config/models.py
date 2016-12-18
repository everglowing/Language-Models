# Configuration of various language models

models = {
    "partial_brnn": {
        "module": "partial_brnn.model",
        "generator": "partial_brnn",
        "processor": "default_process",
        "eval_processor": "partial_brnn",
        "extra_args": ["seq_length", "back_steps"],
        "data_loader": ["vocab"],
        "summary": ""
    }
}