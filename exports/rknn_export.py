from rknn.api import RKNN

def tensorflow_export_rknn_model():
    # Create RKNN object
    rknn = RKNN(verbose=False, verbose_file='./speech_command_build.log')
   
    # Config for Model Input PreProcess
    #rknn.config(quantized_dtype='dynamic_fixed_point-8')
    #rknn.config(quantized_dtype='asymmetric_quantized-u8')
    rknn.config(target_platform=['rv1126'])
    # Load TensorFlow Model
    print('--> Loading model')
    rknn.load_tensorflow(tf_pb='./my_frozen_graph.pb',
                         inputs=['Reshape'],
                         outputs=['labels_softmax'],
                         input_size_list=[[1,3920]])
    print('done')
   
    # Build Model
    print('--> Building model')
    rknn.build(do_quantization=False, dataset='./dataset.txt', pre_compile=False)
    print('done')
   
    # Export RKNN Model
    #rknn.export_rknn('./speech_command_quantized.rknn')
    rknn.export_rknn('./speech_command.rknn')
   
    #import time
    #time.sleep(100)

def export_rknn_model():
    # Create RKNN Object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(target_platform=['rv1126'])
    print('done')
    
    # Load keras model
    print('--> Loading model')
    ret = rknn.load_keras(model=KERAS_MODEL_PATH)
    if ret != 0:
        print('Load keras model failed !')
        exit(ret)
    print('done')
    
    # Build Model # dataset 
    print('--> Building model')    
    rknn.build(do_quantization=False, dataset='./dataset.txt', pre_compile=False)
    print('done')
    
    # Export RKNN Model
    print('--> Export model ')    
    rknn.export_rknn(RKNN_MODEL_PATH)
    print('done')
   
def inference_rknn_model(input_data):
    # Create RKNN Object

    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    #rknn.config(target_platform=['rv1126'])
    print('done')

    print('--> Load RKNN model ')    
    ret = rknn.load_rknn(RKNN_MODEL_PATH)
    print(ret)
    print('done')

    rknn.init_runtime(perf_debug=False)

    outputs, = rknn.inference(inputs=input_data, data_type='float32')
    print(outputs)