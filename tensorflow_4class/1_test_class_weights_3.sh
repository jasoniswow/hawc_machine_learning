# Run different trainings

for b in 0 1 2 3 4 5; do
  for cw in 3; do
    python tf_dnn_multilabel_4class.py --bin $b --weightedClasses ${cw}
  done
done


