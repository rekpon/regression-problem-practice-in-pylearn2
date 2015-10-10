pylearn2で回帰問題の学習経過を見る
====

## Overview
pylearn2で分類を行うチュートリアルは割とあるけど
回帰問題の例はなかったのでつくった
ついでに学習過程を何となく分かった気になれるよう動画で見れるようにした

## Description
* train.py
モデルや学習アルゴリズムの設定と学習を行う

* myextension.py
学習経過をmatplotlibを使って動画にする
mp4に出力する場合は要ffmpeg

## Requirement
* python3
* pylearn2
* ffmpeg (学習経過の動画をmp4で保存するなら)

## Usage
python train.py <options>

- options


## Note

## License
Copyright &copy; 2015 rekpon
Licensed under the [MIT License][mit].

[MIT]: http://www.opensource.org/licenses/mit-license.php