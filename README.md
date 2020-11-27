# Visual-Recognition-HW2

## Requirement
* python 3.6
* tensorflow 2.0.0-beta1
* keras 1.2.2
* opencv 3.3.0
* Etc.

## Dataset
* https://drive.google.com/drive/folders/1Ob5oT9Lcmz7g5mVOcYH3QugA7tV3WsSl
* change the test dataset's name to "test_data"

## Digits Detection
* The pretained weight is in /configs/svhn/weight.h5
* Run digits detection through the following command.
  * `python pred.py -c configs/svhn.json -i test_data/`
