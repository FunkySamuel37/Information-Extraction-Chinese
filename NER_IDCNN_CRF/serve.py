import tensorflow as tf
import os
# from collections import OrderedDict
import pickle

with open('maps.pkl', "rb") as f:
    char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)


SAVE_PATH = './ckpt'
MODEL_NAME = 'ner.ckpt'
VERSION = 1
SERVE_PATH = './serve/{}/{}'.format(MODEL_NAME, VERSION)

checkpoint = tf.train.latest_checkpoint(SAVE_PATH)

tf.reset_default_graph()

config = {}

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(SAVE_PATH)
    saver = tf.train.import_meta_graph(checkpoint + '.meta')
    saver.restore(sess, ckpt.model_checkpoint_path)
    graph = tf.get_default_graph()
    # sess.run(tf.global_variables_initializer())
    char_input = graph.get_tensor_by_name('ChatInputs:0')
    seg_input = graph.get_tensor_by_name('SegInputs:0')
    dropout_input = graph.get_tensor_by_name('Dropout:0')
    # embed_input = graph.get_tensor_by_name('char_embedding/embed_input:0')

    # project/logits/pred:0
    length_output = graph.get_tensor_by_name('Length:0')
    reshape_output = graph.get_tensor_by_name('project/reshape_pred:0')
    transitions_output = graph.get_tensor_by_name('crf_loss/transitions:0')

    # build tensorinfo

    char_input_info = tf.saved_model.utils.build_tensor_info(char_input)
    seg_input_info = tf.saved_model.utils.build_tensor_info(seg_input)
    dropout_input_info = tf.saved_model.utils.build_tensor_info(dropout_input)
    # embed_input_info = tf.saved_model.utils.build_tensor_info(embed_input)

    length_output_info = tf.saved_model.utils.build_tensor_info(length_output)
    reshape_output_info = tf.saved_model.utils.build_tensor_info(reshape_output)
    transitions_output_info = tf.saved_model.utils.build_tensor_info(transitions_output)

    signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            'char_input': char_input_info,
            'seg_input': seg_input_info,
            'dropout_input': dropout_input_info,
            # 'embed_input': embed_input_info,
        },
        outputs={
            'length_output': length_output_info,
            'reshape_output': reshape_output_info,
            'transitions_output': transitions_output_info
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder = tf.saved_model.builder.SavedModelBuilder(SERVE_PATH)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_definition
        })
    builder.save()

