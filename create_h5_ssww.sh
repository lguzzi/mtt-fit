SAMPLE="DY_amc_incl"
HHmin="550"
HHmax="5500"
oddNumber="True"
eventNumber="Odd Even"
for evn in $eventNumber
do
    if [ $evn == "Even" ]
     then
        oddNumber="False"
    fi
    for sam in $SAMPLE 
    do   
        echo $sam
        python3 create_h5.py  --input \
        SKIM_2017/output_${sam}.root \
        --train_odd $oddNumber \
        --min $HHmin \
        --max $HHmax \
        --output output_SKIM2017_${sam}_HHmin_${HHmin}_HHmax_${HHmax}_train${evn}_v41.h5 \
        --features models/FCSetup_v41.py
    done
done


#SAMPLE="ggF_BSM_m300-500 ggF_BSM_m600-1750 TTsem DY_2j"