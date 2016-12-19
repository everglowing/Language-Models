# List of all string messages emitted by the utils

ERRORS = {
    0: "Not enough data. Make seq_length and batch_size small.",
    1: "Input file not found at path {path}",
    2: "{init} must be a a path",
    3: "config.pkl file does not exist in path {init}",
    4: "chars_vocab.pkl file does not exist in path {}",
    5: "No checkpoint found",
    6: "No model path found in checkpoint",
    7: "Command line argument and saved model disagree on '{arg}'",
    8: "Data and loaded model disagree on dictionary mappings!",
    9: "Training model not found for evaluation"
}

LOGS = {
    0: "Reading text file..",
    1: "Reading pre-processed data..",
    2: "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}",
    3: "model saved to {}",
    4: "Evaluation data loaded."
}

FILES = {
    0: "vocab.pkl",
    1: "data.npy",
    2: "config.pkl",
    3: "chars_vocab.pkl",
    4: "args_summary.txt",
    5: "plot_data.csv"
}

