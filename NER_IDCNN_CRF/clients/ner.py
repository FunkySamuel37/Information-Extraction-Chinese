import jieba
import pickle
import numpy as np

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
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)


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

def get_ner_result(client, line):
  with open('../maps.pkl', "rb") as f:
      char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

  _, chars, segs, tags = input_from_line(line, char_to_id)

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

  prediction = client.predict(req_data, request_timeout=10)
  return prediction
