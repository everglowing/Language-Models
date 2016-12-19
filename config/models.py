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
        "summary": "A special BRNN architecture with fixed number of back steps for each timestep."
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
        "summary": "A complete BRNN which doesn't see the next element, and hence can be used as a character predictor."
    },
    "phones_rnn": {
        "module": "models.phones_rnn.model",
        # used to generate batches from the tensor
        "generator": {
            "function": "phones_rnn",
            "extra": {
                "extra_args": [],
                "data_loader": ["ipa_data"],
            }
        },
        "processor": "ipa_process",
        "eval_processor":{
            "function": "phones_rnn",
            "extra": {
                "extra_args": []
            }
        },
        "summary": "An RNN used to learn the mapping between phones and characters"
    },
    "phone_to_phone": {
        "module": "models.phone_to_phone.model",
        # used to generate batches from the tensor
        "generator": {
            "function": "phone_to_phone",
            "extra": {
                "extra_args": [],
                "data_loader": ["ipa_data"],
            }
        },
        "processor": "ipa_process",
        "eval_processor":{
            "function": "phone_to_phone",
            "extra": {
                "extra_args": []
            }
        },
        "summary": "An RNN used to learn the mapping between phones and phones"
    },
}