from tensorflow.python.framework import graph_io
from keras.models import load_model
from keras import backend as K
import tensorflow as tf


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


"""----------------------------------配置路径-----------------------------------"""
h5_model_path = './data/unet_membrane.hdf5'
output_path = "./data"
pb_model_name = "unet_membrane.pb"

"""----------------------------------导入keras模型------------------------------"""
K.set_learning_phase(0)
net_model = load_model(h5_model_path)

print('input is :', net_model.input.name)
print('output is:', net_model.output.name)

"""----------------------------------保存为.pb格式------------------------------"""
sess = K.get_session()
frozen_graph = freeze_session(K.get_session(), output_names=[net_model.output.op.name])
graph_io.write_graph(frozen_graph, output_path, pb_model_name, as_text=False)
