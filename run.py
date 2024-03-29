import os
import sys
import json
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', type=str, default='torch', help='use torch or tf ')
    
    parser.add_argument('--model_name', type=str, default='efficientnet', help='use model name')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--save_path', type=str, default='saved_models/', help='dataset path')
    parser.add_argument('--export_onnx', type=bool, default=False, help='save weight export format onnx')
    parser.add_argument('--onnx_path', type=str, default='saved_models/onnx/', help='save weight export format onnx path')
    parser.add_argument('--export_rknn', type=bool, default=False, help='save weight export format rknn')
    parser.add_argument('--rknn_path', type=str, default='saved_models/rknn/', help='save weight export format rknn path')
    parser.add_argument('--dataset_name', type=str, default='audioset', help='use dataset name urban_sound audioset esc-50')
    parser.add_argument('--dataset_path', type=str, default='', help='dataset path')

    parser.add_argument('--wanted_label', type=str, default='', help='class label list')
    parser.add_argument('--single-cls', action='store_false', help='train multi-class data as single-class')
    parser.add_argument('--num_epochs', type=int, default=50, help='epochs defulat: 5')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size default: 1')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer function')

    parser.add_argument('--pre_train', type=bool, default=False, help='pre train or fine-tune ')
    parser.add_argument('--mixup', type=float, default=0, help="how many (0-1) samples need to be mixup during training")
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
    parser.add_argument('--timem', help='time mask max length', type=int, default=0)
    parser.add_argument("--att_head", type=int, default=4, help="number of attention heads")
    parser.add_argument('--bal', help='if use balance sampling', type=bool, default=False)

    opt = parser.parse_args()

    if opt.framework == 'torch':
        print('use torch')
        from train_torch import train, build_dataset
        sys.path.append('./torch/model')
        #from ensemble import EnsembleModel
        import torch
        from efficient import EffNetAttention
        from torchinfo import summary
    elif opt.framework == 'tensorflow':
        print('use tensorflow ')
        from train_tf import train, build_dataset
    else:
        print("none")

    opt.export_onnx = True

    opt.dataset_path = '/home/ysr/project/dataset/audio/dcase2020_task2/train/fan/'
    train_generator = build_dataset(opt)
    input_shape = (998, 128)
    hidden_dims = [128, 128, 64, 64, 32]
    model = EffNetAttention(input_shape = (1, 998, 128),label_dim=200, b=2, pretrain=False, head_num=0)
    #model = EnsembleModel(input_shape, hidden_dims)

    model.load_state_dict(torch.load("weights/EffNetMean.pth"))
    summary(model, input_size=(1, 998, 128))

    new_state_dict = {}
    for name,param in model.state_dict.items():
        if not name.startswith('attention'):
            new_state_dict[name] = param

    new_model = EffNetAttention(input_shape = (1, 998, 128),label_dim = 16, b=2, pretrain=False, head_num=0, activation = 'softmax')

    for name,param in new_model.state_dict().items():
        if name.startswith('attention'):
            new_state_dict[name] = param

    new_model.load_state_dict(new_state_dict)

    for name, param in new_model.named_parameters():
        if not name.startswith('attention'):
            param.requires_grad = False


    train(opt, new_model, train_generator)

    #if opt.path ==
    #opt.path = '/home/ysr/dataset/audio/' + opt.dataset_name + '/'

    #if opt.dataset_name == 'mine':
    #    opt.pre_train = False
    
    #train_generator, valid_generator, labels, n_classes = build_dataset(opt)
    #opt.n_classes = n_classes

    #model = build_mode(opt)
    #train(opt, model, train_generator, valid_generator)
    
