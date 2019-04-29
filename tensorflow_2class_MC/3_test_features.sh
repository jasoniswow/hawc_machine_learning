# Run different trainings

for b in 2 3 4 5; do
  for f in 0 9; do
    python tf_dnn_multilabel_2class.py --bin $b --featureGroup $f
  done
done


