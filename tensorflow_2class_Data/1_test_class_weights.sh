# Run different trainings

for b in 1; do
  for cw in 1 2 3; do
    python tf_dnn_multilabel_2class.py --bin $b --weightedClasses ${cw}
  done
done


