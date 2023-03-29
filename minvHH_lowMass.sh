python3 minvHH.py \
-i output_SKIM2017_DY_2j_HHmin_0_HHmax_550_train_v41.h5 \
output_SKIM2017_ggF_BSM_m300-500_HHmin_0_HHmax_550_train_v41.h5 \
output_SKIM2017_ggF_BSM_m600-1750_HHmin_0_HHmax_550_train_v41.h5 \
output_SKIM2017_TTsem_HHmin_0_HHmax_550_train_v41.h5 \
-s models/FCSetup_v41.py \
-m neutriniP4_ggF_DY2j_TTsem_v41_alpha1E-6_beta1_Patience100_HHKinMass_0-550_train \
-o massSelection -d True 


