#!/bin/bash

#SBATCH --job-name=teastd_combeff0
#SBATCH --ntasks=1
#SBATCH --nodelist=s3ls2002
##SBATCH --gres=gpu:titan:1
#SBATCH --gres=gpu:2080:2
####SBATCH --gres=gpu:5

source /opt/anaconda3/etc/profile.d/conda.sh

module load use.storage
module load anaconda3

#conda activate tensor2_01 #HPC1  : no multihead
#conda activate tensor2 #HPC1     : Conda tensor2:   Eff net, multihead

#conda activate mix_01  #HPC2: no multihead
conda activate py39ten2 #HPC2: Conda tensor2:   Eff net, multihead

export HDF5_USE_FILE_LOCKING=FALSE
export PATH="/usr/lib/x86_64-linux-gnu/:$PATH"


#------------------ Test new idea from CAM
#python step04_trainer_tch_std.py --out_dir "./11_tch_std"   \
#	                        --train_dir "./../01_split/fold_02_rotate/train" \
#	                        --eva_dir "./../01_split/fold_01/test"\
#			       	--train_emb_dir "./../05_TL_Aug_Att/10_EffB4_pretrain_att_mul_rot2/03_train_emb/" \
#			       	--eva_emb_dir "./../05_TL_Aug_Att/10_EffB4_pretrain_att_mul_rot2/03_test_emb/" \
#				--is_training "yes"  \
#				--is_testing "yes" \
#				--is_extract "yes"
#

#python step04_trainer_tch_std_den11.py --out_dir "./11_tch_std_den11"   \
#	                               --train_dir "./../01_split/fold_02_rotate/train" \
#	                               --eva_dir "./../01_split/fold_01/test"\
#			       	       --train_emb_dir "./../10_teacher_only/10_Den121/03_train_emb/" \
#			       	       --eva_emb_dir "./../10_teacher_only/10_Den121/03_test_emb/" \
#				       --is_training "yes"  \
#				       --is_testing "yes" \
#				       --is_extract "yes"
#

#python step04_trainer_tch_std_eff0.py --out_dir "./11_tch_std_test00"   \
#	                               --train_dir "./../01_split/fold_02_rotate/train" \
#	                               --eva_dir "./../01_split/fold_01/test"\
#			       	       --train_emb_dir "./../10_teacher_only/11_EFF0_DEN121_best/03_train_emb/" \
#			       	       --eva_emb_dir "./../10_teacher_only/11_EFF0_DEN121_best/03_test_emb/" \
#				       --is_training "yes"  \
#				       --is_testing "yes" \
#				       --is_extract "yes"

#-----------------------------------------------------------------------------------
python step04_trainer_tch_std_comb_eff0_b3.py --out_dir "./11_tch_std_comb3_eff0_b3"   \
	                                   --train_dir "./../01_split/fold_02_rotate/train" \
	                                   --eva_dir "./../01_split/fold_01/test"\
			       	           --train_emb_dir "./../21_extract_embedding/10_combine3/03_train_emb/" \
			       	           --eva_emb_dir "./../21_extract_embedding/10_combine3/03_test_emb/" \
				           --is_training "yes"  \
				           --is_testing "yes" \
				           --is_extract "yes"

#python step04_trainer_tch_std_comb_eff0_b4.py --out_dir "./11_tch_std_comb3_eff0_b4"   \
#	                                   --train_dir "./../01_split/fold_02_rotate/train" \
#	                                   --eva_dir "./../01_split/fold_01/test"\
#			       	           --train_emb_dir "./../21_extract_embedding/10_combine3/03_train_emb/" \
#			       	           --eva_emb_dir "./../21_extract_embedding/10_combine3/03_test_emb/" \
#				           --is_training "yes"  \
#				           --is_testing "yes" \
#				           --is_extract "yes"
#
#python step04_trainer_tch_std_comb_eff0_b5.py --out_dir "./11_tch_std_comb3_eff0_b5"   \
#	                                   --train_dir "./../01_split/fold_02_rotate/train" \
#	                                   --eva_dir "./../01_split/fold_01/test"\
#			       	           --train_emb_dir "./../21_extract_embedding/10_combine3/03_train_emb/" \
#			       	           --eva_emb_dir "./../21_extract_embedding/10_combine3/03_test_emb/" \
#				           --is_training "yes"  \
#				           --is_testing "yes" \
#				           --is_extract "yes"
#

#python step04_trainer_tch_std_comb_eff0_b6.py --out_dir "./11_tch_std_comb3_eff0_b6"   \
#	                                   --train_dir "./../01_split/fold_02_rotate/train" \
#	                                   --eva_dir "./../01_split/fold_01/test"\
#			       	           --train_emb_dir "./../21_extract_embedding/10_combine3/03_train_emb/" \
#			       	           --eva_emb_dir "./../21_extract_embedding/10_combine3/03_test_emb/" \
#				           --is_training "yes"  \
#				           --is_testing "yes" \
#				           --is_extract "yes"
#

#python step04_trainer_tch_std_comb_eff0_full.py --out_dir "./11_tch_std_comb3_eff0_full"   \
#	                                   --train_dir "./../01_split/fold_02_rotate/train" \
#	                                   --eva_dir "./../01_split/fold_01/test"\
#			       	           --train_emb_dir "./../21_extract_embedding/10_combine3/03_train_emb/" \
#			       	           --eva_emb_dir "./../21_extract_embedding/10_combine3/03_test_emb/" \
#				           --is_training "yes"  \
#				           --is_testing "yes" \
#				           --is_extract "yes"
#

#python step04_trainer_tch_std_comb_eff0_full_conv.py --out_dir "./11_tch_std_comb3_eff0_full_conv"   \
#	                                   --train_dir "./../01_split/fold_02_rotate/train" \
#	                                   --eva_dir "./../01_split/fold_01/test"\
#			       	           --train_emb_dir "./../10_teacher_only_02/10_ConvLarge/03_train_emb/" \
#			       	           --eva_emb_dir "./../10_teacher_only_02/10_ConvLarge/03_test_emb/" \
#				           --is_training "yes"  \
#				           --is_testing "yes" \
#				           --is_extract "yes"
