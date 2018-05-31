# Run different trainings

for b in 1; do
  for hl in 1; do
    for f in 2; do
      python tf_dnn_multilabel_4class.py --bin $b --hiddenlayer ${hl} --neuronfactor ${f}
    done
  done
done

for b in 1; do
  for hl in 2; do
    for f in 1 2; do
      python tf_dnn_multilabel_4class.py --bin $b --hiddenlayer ${hl} --neuronfactor ${f}
    done
  done
done

