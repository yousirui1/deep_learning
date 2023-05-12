import os
import tensorflow.compat.v2 as tf
from tensorflow.python.framework import graph_util, graph_io

## pb 转 h5 tf2.0
def tf2keras(pb_path, h5_path):
    model = tf.keras.models.load_model(pb_path)
    tf.keras.models.save_model(model, h5_path, save_format='h5')
    model = tf.keras.models.load_model(h5_path)
    model.summary()


def keras2tf(h5_path, pb_path):
    print("h5_path", h5_path);
    model = tf.keras.models.load_model(h5_path)
    tf.keras.models.save_model(model, pb_path, save_format='pb')

   


def test(h5_path, pb_path):
    if os.path.exists(pb_path) == False:
        os.mkdir(pb_path)
    out_nodes = []
    h5_model = tf.keras.models.load_model(h5_path)
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))
    sess = backend.get_session()

    # 写入pb模型文件
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, pb_path, name=model_name, as_text=False)


