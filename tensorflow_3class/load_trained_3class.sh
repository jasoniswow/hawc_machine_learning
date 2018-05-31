# Run different trainings

bin=1

for f in -1 0 1 2 3 4 5 6 9; do

    echo "************"
    echo $f
    python load_model_from_text_3class.py --modelFile 3_test_features/Bin_${bin}_features_${f}_weight_11_HL_1_NF_1_LR_0.001_epochs_5000 --featureGroup $f --bin ${bin}

done





