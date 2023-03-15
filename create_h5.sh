SAMPLES="300 350 400 450 500 600 800 1000 1250 1500 1750"
#SAMPLES="output_ggF_BSM_lowMass.root"
#SAMPLES="ggF_BSM"
#SAMPLES="ggF_BSM/SKIM_ggF_BulkGraviton_m*"
for sam in $SAMPLES
do
    echo "Now creating "$sam" h5 file "
    python3 create_h5.py  --input \
    ../SKIM_ggF_BSM/output_ggF_BSM_m${sam}.root \
    --train_odd True --output output_ggF_BSM_m${sam}_trainOdd_v31.h5 --features models/FCSetup_v31.py
done
