import jieba
import pickle
import numpy as np
from tensorflow.contrib.crf import viterbi_decode

def get_seg_features(string):
    """
    Segment text with jieba
    features are represented in bies format
    s donates single word
    """
    seg_feature = []

    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature


def full_to_half(s):
    """
    Convert full-width character to half-width one
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


def replace_html(s):
    s = s.replace('&quot;', '"')
    s = s.replace('&amp;', '&')
    s = s.replace('&lt;', '<')
    s = s.replace('&gt;', '>')
    s = s.replace('&nbsp;', ' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;", "")
    s = s.replace("\xa0", " ")
    return (s)


def input_from_line(line, char_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                    for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    return inputs


def decode(logits, lengths, matrix, num_tags):
    """
    :param logits: [batch_size, num_steps, num_tags]float32, logits
    :param lengths: [batch_size]int32, real length of each sequence
    :param matrix: transaction matrix for inference
    :return:
    """
    # inference final labels usa viterbi Algorithm
    paths = []
    small = -1000.0
    start = np.asarray([[small]*num_tags +[0]])
    for score, length in zip(logits, lengths):
        score = score[:length]
        pad = small * np.ones([length, 1])
        logits = np.concatenate([score, pad], axis=1)
        logits = np.concatenate([start, logits], axis=0)
        path, _ = viterbi_decode(logits, matrix)

        paths.append(path[1:])
    return paths

def result_to_json(string, tags):
    item = {"text": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item

def get_ner_result(client, line):
    with open('maps.pkl', "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    num_tags = len(tag_to_id)
    inputs = input_from_line(line, char_to_id)
    _, chars, segs, tags = inputs

    req_data = [{
        'in_tensor_name': 'char_input',
        'in_tensor_dtype': 'DT_INT32',
        'data': np.asarray(chars)
    }, {
        'in_tensor_name': 'seg_input',
        'in_tensor_dtype': 'DT_INT32',
        'data': np.asarray(segs)
    }, {
        'in_tensor_name': 'dropout_input',
        'in_tensor_dtype': 'DT_FLOAT',
        'data': 1.0
    }]
    # , {
    #     'in_tensor_name': 'embed_input',
    #     'in_tensor_dtype': 'DT_FLOAT',
    #     'data':
    # }

    prediction = client.predict(req_data, request_timeout=10)
    length_output = [prediction['length_output']]
    transitions_output = prediction['transitions_output']
    reshape_output = prediction['reshape_output']
    # print(prediction)

    batch_paths = decode(logits=reshape_output, lengths=length_output, matrix=transitions_output, num_tags=num_tags)
    print('batch_paths')
    print(batch_paths)
    tags = [id_to_tag[idx] for idx in batch_paths[0]]
    print(id_to_tag)
    return result_to_json(inputs[0][0], tags)
    # return prediction
