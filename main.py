"""
This project implements an improvement of the loss function based on the prediction vector perspective, as well as an
improvement of the transformer using feature fusion gating.
This project achieves better results in RUL prediction

Author: Zhou Zhihao,Harbin Institute of Technology
Email: 22S002045@stu.hit.edu.cn
Date: 30/6/2024


"""
import os
import argparse
from data_load import *
from model import *
from visualize import *
from torch.utils.data import DataLoader
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")
torch.manual_seed(1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":

#------PARAMETER DEFINITION------#
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FD002', help='which dataset to run, FD001-FD004')
    parser.add_argument('--modes', type=str, default='test', help='train or test')
    parser.add_argument('--path', type=str, default='/home/ps/code/zhouzhihao/FFG-Teans-test/saved_model/model/-FD002_ffg_tansformer.pth')
    parser.add_argument('--smooth_param', type=float, default=0.2, help='none or freq')
    parser.add_argument('--train_seq_len', type=int, default=30, help='train_seq_len')
    parser.add_argument('--test_seq_len', type=int, default=30, help='test_seq_len')
    opt = parser.parse_args()
    print(opt)

    if opt.modes == "train":
        print('********Sorry, the training program has not been uploaded yet !! ************')

    elif opt.modes == "test":
        print('--------------Testing begins------------')

        PATH = opt.path
        print(PATH)
        group_train, y_test, group_test, X_test = data_processing(opt.dataset,opt.smooth_param)
        test_dataset = SequenceDataset(mode='test',group = group_test, y_label=y_test, sequence_train=opt.train_seq_len, patch_size=opt.train_seq_len)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        model = AUM_Transformer(seq_size=opt.train_seq_len, patch_size=3, in_chans=14,embed_dim=64, depth=1,
                                num_heads=4,decoder_embed_dim=64, decoder_depth=1, decoder_num_heads=4,
                                norm_layer=nn.LayerNorm,batch_size = 30)
        model.load_state_dict(torch.load(PATH))

        if torch.cuda.is_available():
            model = model.to(device)

        model.eval()
        result=[]
        mse_loss=0

        with torch.no_grad():
            test_epoch_loss = 0
            for X,y in test_loader:
                if torch.cuda.is_available():
                    X=X.cuda()
                    y=y.cuda()

                y_hat_recons = model.forward(X)

                y_hat_unscale = y_hat_recons[0]*125
                result.append(y_hat_unscale.item())

        y_test.index = y_test.index
        result = y_test.join(pd.DataFrame(result))
        # result.to_csv(opt.dataset+'_result.csv')
        result['RUL'].clip(upper=125, inplace=True)

        error = result.iloc[:,1]-result.iloc[:,0]
        res=0
        for value in error:
            if value < 0:
                res = res + np.exp(-value / 13) - 1
            else:
                res = res + np.exp(value / 10) - 1

        rmse =  np.sqrt(np.mean(error ** 2))

        print("testing score: %1.5f" % (res))
        print("testing rmse: %1.5f" % (rmse))

        result = result.sort_values('RUL', ascending=False)
        # visualize the testing result
        visualize(result, rmse)

        count = np.sum(error > 0)
        # print("Number of elements greater than 0:", count)
        from thop import profile
        flops,params = profile(model,(X,))
        print("Total FLOPs2:", flops)
        print("Total params:", params)