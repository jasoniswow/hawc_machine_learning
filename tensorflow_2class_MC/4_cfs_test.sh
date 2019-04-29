# Run different trainings

for cfs in 080 090 100 110 120 130 140 150; do
  mkdir cfs_test_${cfs}

  for b in 0 1 2 3 4 5; do
    for f in 9; do
      python tf_dnn_multilabel_2class.py --bin $b --featureGroup $f --coreposition ${cfs}
    done
  done
  mv Bin* cfs_test_${cfs}/

done
































