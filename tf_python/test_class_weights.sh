# Run different trainings

for cw in 0 1 2 3; do
    python tf_dnn_multilabel.py --bin 2 --weightedClasses ${cw} --hiddenlayer 1 --epochs 50000 --displayStep 500
done


