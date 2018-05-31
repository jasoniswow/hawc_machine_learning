# Run different trainings

for b in 0 1 2 3 4 5; do
  for cw in 1; do
    python tf_dnn_multilabel_3class.py --bin $b --weightedClasses ${cw}
  done
done


