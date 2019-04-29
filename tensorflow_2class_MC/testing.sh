#

for b in 0 1 2 3 4 5; do

  python test_dnn_multilabel_2class.py --bin $b --featureGroup 9 --weightedClasses 1 --hiddenlayer 0 --neuronfactor 1 --learningRate 0.01 --epochs 1000 --displayStep 10

done



