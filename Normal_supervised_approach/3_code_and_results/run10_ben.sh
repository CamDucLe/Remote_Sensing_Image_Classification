#!/bin/bash

#SBATCH --job-name=im_sel
#SBATCH --ntasks=1
#SBATCH --nodelist=s3ls2001
##SBATCH --gres=gpu:titan:2
#SBATCH --gres=gpu:2080:4

source /opt/anaconda3/etc/profile.d/conda.sh

module load use.storage
module load anaconda3

#conda activate tensor2_01 #HPC1  : no multihead
conda activate tensor2 #HPC1     : Conda tensor2:   Eff net, multihead

#conda activate mix_01  #HPC2: no multihead
#conda activate py39ten2 #HPC2: Conda tensor2:   Eff net, multihead

export HDF5_USE_FILE_LOCKING=FALSE
export PATH="/usr/lib/x86_64-linux-gnu/:$PATH"


#------------------ Test new idea 
#python step10_nasmobile.py --out_dir "./10_NASmobile"   --train_dir "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "yes"  --is_testing "yes" --is_extract "yes"

#------------------------------
#python step10_effb0_att_05.py --out_dir "./10_EffB0_Att_05"   --train_dir "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "yes"  --is_testing "yes" --is_extract "yes"

