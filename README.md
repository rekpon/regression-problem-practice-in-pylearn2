pylearn2で回帰問題の学習経過を見る
====

## Overview
pylearn2で分類問題を行うチュートリアルは割とあるけど回帰問題の例はなかったのでつくった  
ついでに学習過程を何となく分かった気になれるよう動画で見れるようにした  

![学習経過](/course.gif)
こんな感じに学習する  
(青線 : 教師データ、黒線 : モデルの予測値、赤線 : モデルの初期予測値)

## Description
基本的には分類問題と同じような事をする  
大雑把にいうと分類先のクラスが離散値から連続値に広がっただけ  

* train.py  
モデルや学習アルゴリズムの設定と学習を行う  

* myextension.py  
学習経過をmatplotlibを使って動画にする  
mp4に出力する場合は要ffmpeg  

学習したモデルはfunkmodel.pklに保存されます  

## Requirement
* python3  
* pylearn2  
* ffmpeg (学習経過の動画をmp4で保存するなら)  

## Usage
    $ python train.py -p  

* options  
- -p : 学習経過を動画で表示  
- -f \<filename\> : -p の動画を\<filename\>.mp4に保存(要ffmpeg)  

## Note
* 学習回数やノード数より学習サンプル数の方増やした方が手軽に結果は良くなる傾向  
* 現状でも動いているので放置しているがが、多分cost関数を変えるべき  
* myextension.pyの記述を少し変えると動画を上のようにgifでも保存できます(要ImageMagick)  

## License
Copyright &copy; 2015 rekpon  
Licensed under the [MIT License][mit].  

[MIT]: http://www.opensource.org/licenses/mit-license.php
