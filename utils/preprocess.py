# Set of preprocessors which are added as flags
from .strings import LOGS
from random import randint

# Helper functions
def count_words(sentence):
    return len(sentence.split())

def sentence_sort(data):
    if u'\u0964' in data:
        # Special case of hindi
        separator = u'\u0964'
    else:
        separator = '.'
    print(LOGS[7].format(separator))
    # Remove all the new line characters
    data.replace('\n', separator)
    str_list = data.split(separator)
    # Remove empty sentences
    str_list = filter(None, str_list)
    str_list.sort(key=count_words)
    # Join the string inserting \n randomly
    output = ""
    last_inserted = -1
    for index, sentence in enumerate(str_list):
        if len(sentence.split()) < 4:
            last_inserted = index
            continue
        output += sentence + separator
        # Logic for inserting new lines
        num = randint(4, 10)
        if num < index - last_inserted:
            output += "\n"
            last_inserted = index
    return output
