# Run different trainings

bin=2
hl=4
lr=0.0001
nf=2.5

python load_resume_training.py --modelDir Bin_${bin}_features_9_weight_11_HL_${hl}_NF_${nf}_LR_${lr}_epochs_50000 --bin ${bin} --hiddenlayer ${hl} --neuronfactor ${nf} --learningRate ${lr} --epochs 100 --displayStep 1

