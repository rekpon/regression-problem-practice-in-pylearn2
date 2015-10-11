#coding: utf-8
""" pylearn2 でsin関数を近似するサンプルプログラム。

[実行方法]
$ python train.py

[オプション]
-p : epoch毎にモデルの予測値を保存、学習終了後にアニメーションで遷移を表示。

-f, --file <finename> : -pオプションのアニメーションをmp4ファイルで保存(要ffmpeg)

[出力]
学習結果は ./funcmodel.pkl に保存。

[note]
1. MAX_EPOCH は学習データを繰り返し学習する回数。
   SIZE_DATA は作成する学習データの個数。

2. 近似する関数を変えるにはUnknownFunk()のX,yを変更。
   Xが入力値、yが出力値の2-dimention nparray。

3. 各隠れ層のノード数や層数によりtrainerのlearning_rateの調節が必要。
"""

MAX_EPOCHS = 100
SIZE_DATA = 200

import random
import numpy as np
import pylearn2
import time
from math import pi, cos, sin
from pylearn2.termination_criteria import EpochCounter
from pylearn2.training_algorithms import sgd
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.models import mlp
from pylearn2.train import Train
from optparse import OptionParser
from myextension import PlotPredictionOnMonitor

# 学習させる未知の関数の入出力ペアを生成
class sinDataset(DenseDesignMatrix):
    def __init__(self, n):
        X = []
        y = []
        for i in range(0, n):
            x = random.uniform(-0.5, 2*pi+0.5)
            X.append([x])
            y.append([sin(x)])

        X = np.array(X)
        y = np.array(y)
        super(sinDataset, self).__init__(X=X, y=y)

def main():
    start_time = time.clock()

    # optin parser
    parser = OptionParser()
    parser.add_option("-p", dest="plot_prediction",
                      action="store_true", default=False,
                      help="plot model prediction transitions")
    parser.add_option("-f", "--file", dest="out_filename", default=None,
                      help="write animation to FILE (require -a option)", metavar="FILE")
    (options, args) = parser.parse_args()

    # make Detaset
    ds = sinDataset(SIZE_DATA)

    # make layers
    hidden_layer1 = mlp.Tanh(layer_name='hidden1', dim=20, irange=0.5, init_bias=1.0)
    hidden_layer2 = mlp.Tanh(layer_name='hidden2', dim=4, irange=0.5, init_bias=1.0)
    output_layer = mlp.Linear(layer_name='out', dim=1, irange=0.5, init_bias=1)

    # set layers
    layers = [hidden_layer1, hidden_layer2, output_layer]
    model = mlp.MLP(layers, nvis=1)

    # set training rule and extensions
    algorithm = sgd.SGD(
        learning_rate = 0.01,
        batch_size = 1,
        monitoring_batch_size = 1,
        monitoring_batches =  1,
        monitoring_dataset = ds,
        termination_criterion = EpochCounter(MAX_EPOCHS)
    )
    extensions = [sgd.MonitorBasedLRAdjuster()]
    if options.plot_prediction:
        plotEx = PlotPredictionOnMonitor()
        extensions.append(plotEx)

    trainer = Train(model = model,
                    algorithm = algorithm,
                    dataset = ds,
                    extensions = extensions,
                    save_path='./funcmodel.pkl',
                    save_freq=500)

    # training loop
    trainer.main_loop()

    end_time = time.clock()
    print("tortal_seconds_this_learning : %f s(%f min)" % (end_time - start_time, (end_time - start_time)/60))
    if options.plot_prediction:
        plotEx.plot(out_filename=options.out_filename)

if __name__ == '__main__':
    main()
