module.exports = {
  apps : [{
    name      : 'model_server',
    script: 'tensorflow_model_server',
    exec_interprete: 'bash',
    args: '--port=9000 --model_name=ner.ckpt --model_base_path=/home/ubuntu/serve/ner.ckpt --saved_model_tags=serve',
    exec_mode: 'fork',
    log: true,
  }, {
    name: 'predict_client',
    script: 'client.py',
    cwd: '/home/ubuntu/Information-Extraction-Chinese/NER_IDCNN_CRF',
    interpreter: '/usr/bin/python3',
    exec_mode: 'fork',
    log: true,
    env: {'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python'}
  }],
};
