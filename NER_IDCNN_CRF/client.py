# import tensorflow as tf
import argparse
from predict_client.prod_client import ProdClient
from clients.ner import get_ner_result

from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--map_file', '-m', help="a path to map file", type= str)

FLAGS=parser.parse_args()

HOST = '0.0.0.0:9000'
MODEL_NAME = 'ner.ckpt'
MODEL_VERSION = 1

client = ProdClient(HOST, MODEL_NAME, MODEL_VERSION)

# flags = tf.app.flags
# flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")

# FLAGS = tf.app.flags.FLAGS

@app.route("/ner", methods=['POST'])
def get_prediction():
    req_data = request.get_json()
    # raw_data = req_data['data']
    raw_data = req_data['input']
    if raw_data is None:
        return jsonify({ 'error_msg': 'wrong input, input should be string or string array'})

    # prediction = None
    # if isinstance(raw_data, list) is False:
    prediction = get_ner_result(client, raw_data)
    # else:
    #     prediction = map(lambda x: get_ner_result(client, x), raw_data)
    print(raw_data)
    print(prediction)
    # ndarray cannot be converted to JSON
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=3000)
