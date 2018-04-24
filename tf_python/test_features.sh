# Run different trainings

for b in 0 1 2 3 4 5; do
  for f in -1 0 1 2 3 4 9; do
    python tf_dnn_multilabel.py --bin $b --featureGroup $f --epochs 50000 --displayStep 500
  done
done


