
cn=3
wt=doubleH

for b in 0 1 2 3 4 5; do
  python2 mislabeled_${cn}class.py -i /data4/hawcroot/sim/reco/labeled_each_bin/labeled_sweets_dec20_noise_MPF_allParticle_${cn}class_${wt}_bin_${b}.xcd -b $b
done









