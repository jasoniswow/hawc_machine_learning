# Run different trainings

for b in 0 1 2 3 4 5; do
  python tf_dnn_multilabel_2class.py --bin $b
done


