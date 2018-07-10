# import tensorflow as tf
import argparse
from predict_client.prod_client import ProdClient
from clients.ner import get_ner_result

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



line = '香港的房价已经到达历史巅峰,乌溪沙地铁站上盖由新鸿基地产公司开发的银湖天峰,现在的尺价已经超过一万五千港币。'


# print(req_data)
# trans = self.trans.eval()
# lengths, scores = client.predict(req_data, request_timeout=10)
# batch_paths = self.decode(scores, lengths, trans)
prediction = get_ner_result(client, line)
print(prediction)
