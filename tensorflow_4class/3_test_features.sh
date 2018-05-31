# Run different trainings

for b in 1; do
  for f in -1 0 1 2 3 4 5 6 9; do
    python tf_dnn_multilabel_4class.py --bin $b --featureGroup $f
  done
done


