#coding: utf-8
""" pylearn2 用 extensions
[PlotPredictionOnMonitor]
- epoch毎にテスト用データに対するモデルの予測値をプロットし、遷移をアニメーション表示する
- アニメーションの開始には<this_extension>.plot()
"""

import random
import numpy as np
import theano
import pylearn2
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.animation as animation
from math import pi, cos, sin
from pylearn2.train_extensions import TrainExtension

# extend range function
def drange(begin, end, step):
    n = begin
    while n+step < end:
        yield n
        n += step

class PlotPredictionOnMonitor(TrainExtension):
    def setup(self, model, dataset, algorithm):
        self.model = model
        self.algorithm = algorithm

        # figure plot setup
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        ax.grid(True,linestyle='-', color='0.75')
        ax.set_xlim([0, 2*pi])

        ## make test dataset
        self.test_data = []
        for i in drange(0, 6.29, 0.01):
            self.test_data.append([i])
        self.test_data = np.array(self.test_data)

        ## save initial output
        self.frame_list = []
        output_data = self.model.fprop(theano.shared(self.test_data, name='inputs')).eval()
        pylab.plot(self.test_data, output_data, '-', color='r')

    def on_monitor(self, model, dataset, algorithm):
        # plot model output on every epoch
        output_data = self.model.fprop(theano.shared(self.test_data, name='inputs')).eval()
        one_frame = pylab.plot(self.test_data, output_data, '-', color='k')
        self.frame_list.append(one_frame)
        del output_data

    # plot sincurve and start animation
    def plot(self, out_filename=None):
        x = np.arange(0, 2*pi+0.1, 0.1)
        y = np.sin(x)
        pylab.plot(x, y, color='b', label="sin(x)")

        # animation
        print("Pless Enter to start animation")
        input()
        anim = animation.ArtistAnimation(self.fig,self.frame_list,
                                         interval=100,
                                         repeat_delay=10000)
        if out_filename is not None:
            print("saving animation to", out_filename+".mp4", "...")
            anim.save(out_filename+".gif", writer="ffmpeg", fps=24)
            print("done.")
        plt.show()
