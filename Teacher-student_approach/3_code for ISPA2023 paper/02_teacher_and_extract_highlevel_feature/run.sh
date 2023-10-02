#!/bin/bash

#SBATCH --job-name=combine
#SBATCH --ntasks=1
#SBATCH --nodelist=s3ls2001
##SBATCH --gres=gpu:titan:1
#SBATCH --gres=gpu:2080:1

source /opt/anaconda3/etc/profile.d/conda.sh

module load use.storage
module load anaconda3

conda activate rmss01 #HPC2


export HDF5_USE_FILE_LOCKING=FALSE
export PATH="/usr/lib/x86_64-linux-gnu/:$PATH"

#python step10_combine.py  --out_dir "10_combine"   \
#	                  --emb01_train   "./../10_teacher_only_02/10_ConvLarge/03_train_emb/" \
#	                  --emb01_test    "./../10_teacher_only_02/10_ConvLarge/03_test_emb" \
#	                  --emb02_train   "./../10_teacher_only_02/10_Effb7/03_train_emb/" \
#	                  --emb02_test    "./../10_teacher_only_02/10_Effb7/03_test_emb/" \
#	                  --is_training   "yes"  \
#	                  --is_testing    "yes" \
#	                  --is_extract    "yes"  
#
#python step10_combine3.py  --out_dir "10_combine3_02"   \
#	                  --emb01_train   "./../10_teacher_only_02/10_ConvLarge/03_train_emb/" \
#	                  --emb01_test    "./../10_teacher_only_02/10_ConvLarge/03_test_emb" \
#	                  --emb02_train   "./../10_teacher_only_02/10_Effb7/03_train_emb/" \
#	                  --emb02_test    "./../10_teacher_only_02/10_Effb7/03_test_emb/" \
#	                  --emb03_train   "./../10_teacher_only_02/10_Den201/03_train_emb/" \
#	                  --emb03_test    "./../10_teacher_only_02/10_Den201/03_test_emb/" \
#	                  --is_training   "yes"  \
#	                  --is_testing    "yes" \
#	                  --is_extract    "yes"  
#

#python step10_combine3_03.py  --out_dir "10_combine3_03"   \
#	                  --emb01_train   "./../10_teacher_only_02/10_ConvLarge/03_train_emb/" \
#	                  --emb01_test    "./../10_teacher_only_02/10_ConvLarge/03_test_emb" \
#	                  --emb02_train   "./../10_teacher_only_02/10_Effb7/03_train_emb/" \
#	                  --emb02_test    "./../10_teacher_only_02/10_Effb7/03_test_emb/" \
#	                  --emb03_train   "./../10_teacher_only_02/10_Den201/03_train_emb/" \
#	                  --emb03_test    "./../10_teacher_only_02/10_Den201/03_test_emb/" \
#	                  --is_training   "yes"  \
#	                  --is_testing    "yes" \
#	                  --is_extract    "yes"  
#

python step10_combine3_04.py  --out_dir "10_combine3_04"   \
	                  --emb01_train   "./../10_teacher_only_02/10_ConvLarge/03_train_emb/" \
	                  --emb01_test    "./../10_teacher_only_02/10_ConvLarge/03_test_emb" \
	                  --emb02_train   "./../10_teacher_only_02/10_Effb7/03_train_emb/" \
	                  --emb02_test    "./../10_teacher_only_02/10_Effb7/03_test_emb/" \
	                  --emb03_train   "./../10_teacher_only_02/10_Den201/03_train_emb/" \
	                  --emb03_test    "./../10_teacher_only_02/10_Den201/03_test_emb/" \
	                  --is_training   "yes"  \
	                  --is_testing    "yes" \
	                  --is_extract    "yes"  

#python step10_concat.py  --out_dir "10_concat"   \
#	                  --emb01_train   "./../10_teacher_only_02/10_ConvLarge/03_train_emb/" \
#	                  --emb01_test    "./../10_teacher_only_02/10_ConvLarge/03_test_emb" \
#	                  --emb02_train   "./../10_teacher_only_02/10_Effb7/03_train_emb/" \
#	                  --emb02_test    "./../10_teacher_only_02/10_Effb7/03_test_emb/" \
#	                  --is_training   "yes"  \
#	                  --is_testing    "yes" \
#	                  --is_extract    "yes"  
