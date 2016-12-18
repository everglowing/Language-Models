# Configuration of various language models

models = {
    "partial_brnn": {
        "module": "partial_brnn.model",
        "generator": "next_char_char",
        "processor": "default_process"
    }
}