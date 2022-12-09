import os
import numpy as np
import tensorflow as tf

print("TF version:{}".format(tf.__version__))
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

KERAS_MODEL_PATH = "./test.h5"
RKNN_MODEL_PATH = "./test.rknn"
LITE_MODEL_PATH = "./test.tflite"
TENSORFLOW_MODEL_PATH = "./model/audio.pb"
 
def inference_tflite_model(input_data):
    # Create RKNN Object

    print('--> Load TFLite model ')    
    interpreter = tf.lite.Interpreter(model_path=LITE_MODEL_PATH)
    interpreter.allocate_tensors()
    print('done')

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(input_details)
    print(output_details)

    print(input_data.reshape(1,1,2626))

    interpreter.set_tensor(input_details[0]['index'], input_data.reshape(1,1,2626))

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    #end = time.time()

    print(output_data)
 

def export_tensorflow2lite_model(pb_file = TENSORFLOW_MODEL_PATH, lite_file = LITE_MODEL_PATH):
 
    input_tensor_name=["Reshape"]
    input_tensor_shape={"Reshape":[1, 7920]}
 
    output_tensor=["labels_softmax"]
 
    convertr=tf.lite.TFLiteConverter.from_frozen_graph(pb_file,input_arrays=input_tensor_name
                                                   ,output_arrays=output_tensor
                                                   ,input_shapes=input_tensor_shape)
 
    tflite_model=convertr.convert()
 
 
    with open(lite_file,'wb') as f:
        f.write(tflite_model)
 

def export_keras2lite_model(keras_file = KERAS_MODEL_PATH, lite_file = LITE_MODEL_PATH):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
    tflite_model = converter.convert()
    with open(lite_file, 'wb') as f:
        f.write(tflite_model)
    def export_tensorflow2lite_model(pb_file = TENSORFLOW_MODEL_PATH, lite_file = LITE_MODEL_PATH):
 
    input_tensor_name=["Reshape"]
    input_tensor_shape={"Reshape":[1, 7920]}
 
    output_tensor=["labels_softmax"]
 
    convertr=tf.lite.TFLiteConverter.from_frozen_graph(pb_file,input_arrays=input_tensor_name
                                                   ,output_arrays=output_tensor
                                                   ,input_shapes=input_tensor_shape)
 
    tflite_model=convertr.convert()
 
 
    with open(lite_file,'wb') as f:
        f.write(tflite_model)
 
    
def converer_keras_to_tflite_v1_(keras_path, outputs_layer=None, out_tflite=None):
    """
    :param keras_path: keras *.h5 files
    :param outputs_layer
    :param out_tflite: output *.tflite file
    :return:
    """
    model_dir = os.path.dirname(keras_path)
    model_name = os.path.basename(keras_path)[:-len(".h5")]
    # 加载keras模型, 结构打印
    model_keras = tf.keras.models.load_model(keras_path)
    print(model_keras.summary())
    # 从keras模型中提取fc1层, 需先保存成新keras模型, 再转换成tflite
    model_embedding = tf.keras.models.Model(inputs=model_keras.input,
                                            outputs=model_keras.get_layer(outputs_layer).output)
    print(model_embedding.summary())
    keras_file = os.path.join(model_dir, "{}_{}.h5".format(model_name, outputs_layer))
    tf.keras.models.Model.save(model_embedding, keras_file)
 
    # converter = tf.lite.TocoConverter.from_keras_model_file(keras_file)
    converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)  # tf1.3
    # converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)  # tf2.0
    tflite_model = converter.convert()
 
    if not out_tflite:
        out_tflite = os.path.join(model_dir, "{}_{}.tflite".format(model_name, outputs_layer))
    open(out_tflite, "wb").write(tflite_model)
    print("successfully convert to tflite done")
    print("save model at: {}".format(out_tflite))
 
def converer_keras_to_tflite_v1(keras_file = KERAS_MODEL_PATH, lite_file = LITE_MODEL_PATH):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
    tflite_model = converter.convert()
    with open(lite_file, 'wb') as f:
        f.write(tflite_model)
    
def converer_keras_to_tflite_v2(keras_path, outputs_layer=None, out_tflite=None, optimize=False, quantization=False):
    """
    :param keras_path: keras *.h5 files
    :param outputs_layer: default last layer
    :param out_tflite: output *.tflite file
    :param optimize
    :return:
    """
    if not os.path.exists(keras_path):
        raise Exception("Error:{}".format(keras_path))
    model_dir = os.path.dirname(keras_path)
    model_name = os.path.basename(keras_path)[:-len(".h5")]
    # 加载keras模型, 结构打印
    # model = tf.keras.models.load_model(keras_path)
    model = tf.keras.models.load_model(keras_path, custom_objects={'tf': tf}, compile=False)
 
    print(model.summary())
    if outputs_layer:
        # 从keras模型中提取层,转换成tflite
        model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(outputs_layer).output)
        # outputs = [model.output["bbox"],model.output["scores"]]
        # model = tf.keras.models.Model(inputs=model.input, outputs=outputs)
        print(model.summary())
    # converter = tf.lite.TocoConverter.from_keras_model_file(keras_file)
    # converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)  # tf1.3
    converter = tf.lite.TFLiteConverter.from_keras_model(model)  # tf2.0
    prefix = [model_name, outputs_layer]
    # converter.allow_custom_ops = False
    # converter.experimental_new_converter = True
    """"
    https://tensorflow.google.cn/lite/guide/ops_select
    我们优先推荐使用 TFLITE_BUILTINS 转换模型，然后是同时使用 TFLITE_BUILTINS,SELECT_TF_OPS ，
    最后是只使用 SELECT_TF_OPS。同时使用两个选项（也就是 TFLITE_BUILTINS,SELECT_TF_OPS）
    会用 TensorFlow Lite 内置的运算符去转换支持的运算符。
    有些 TensorFlow 运算符 TensorFlow Lite 只支持部分用法，这时可以使用 SELECT_TF_OPS 选项来避免这种局限性。
    """
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
    #                                        tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if optimize:
        print("weight quantization")
        # Enforce full integer quantization for all ops and use int input/output
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        prefix += ["optimize"]
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
 
    if quantization == "int8":
        converter.representative_dataset = representative_dataset_gen
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8  # or tf.uint8
        # converter.inference_output_type = tf.int8  # or tf.uint8
        converter.target_spec.supported_types = [tf.int8]
    elif quantization == "float16":
        converter.target_spec.supported_types = [tf.float16]
 
    prefix += [quantization]
    if not out_tflite:
        prefix = [str(n) for n in prefix if n]
        prefix = "_".join(prefix)
        out_tflite = os.path.join(model_dir, "{}.tflite".format(prefix))
    tflite_model = converter.convert()
    open(out_tflite, "wb").write(tflite_model)
    print("successfully convert to tflite done")
    print("save model at: {}".format(out_tflite))
