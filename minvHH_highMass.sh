python3 minvHH.py \
-i output_SKIM2017_DY_2j_HHmin_550_HHmax_5500_train_v41.h5 \
output_SKIM2017_ggF_BSM_m300-500_HHmin_550_HHmax_5500_train_v41.h5 \
output_SKIM2017_ggF_BSM_m600-1750_HHmin_550_HHmax_5500_train_v41.h5 \
output_SKIM2017_TTsem_HHmin_550_HHmax_5500_train_v41.h5 \
-s models/FCSetup_v41.py \
-m neutriniP4_ggF_DY2j_TTsem_v41_alpha1E-6_beta1_Patience100_HHKinMass_550-5500_train \
-o massSelection -d True 


