# Run different trainings

for hl in 1 2 3 4; do
  for f in 1.0 1.5 2.0 2.5 3.0; do
    python tf_dnn_multilabel.py --bin 2 --hiddenlayer ${hl} --neuronfactor ${f} --learningRate 0.0001 --epochs 50000 --displayStep 500
  done
done

