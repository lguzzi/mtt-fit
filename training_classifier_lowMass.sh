
python3 FC_training_Functional.py \
--input output_SKIM2017_DY_amc_incl_trainEven_v41.h5 \
output_SKIM2017_ggF_BSM_m300_trainEven_v41.h5 \
output_SKIM2017_ggF_BSM_m350_trainEven_v41.h5 \
output_SKIM2017_ggF_BSM_m400_trainEven_v41.h5 \
output_SKIM2017_ggF_BSM_m450_trainEven_v41.h5 \
output_SKIM2017_ggF_BSM_m500_trainEven_v41.h5 \
output_SKIM2017_ggF_BSM_m550_trainEven_v41.h5 \
output_SKIM2017_TTsem_trainEven_v41.h5 \
--output neutriniP4_ggF_m300-550_DY_incl_TTsem_v41_alpha1E-6_beta1_Patience100_trainEven \
--setup models/FCSetup_v41.py \
--cpu 