#!/bin/bash

#SBATCH --job-name=convl
#SBATCH --ntasks=1
#SBATCH --nodelist=s3ls2001
##SBATCH --gres=gpu:titan:2
#SBATCH --gres=gpu:2080:1

source /opt/anaconda3/etc/profile.d/conda.sh

module load use.storage
module load anaconda3


conda activate tensor2 #HPC1  # there is EffNet, multihead for conda tensor2 --> only on HPC1

#conda activate py39ten2 #HPC2
#conda activate rmss01 #HPC2


export HDF5_USE_FILE_LOCKING=FALSE
export PATH="/usr/lib/x86_64-linux-gnu/:$PATH"

#---------- ResNet
#python step10_trainer_res50.py    --out_dir "10_ResNet50"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "yes"  --is_testing "yes" --is_extract "yes"  
#python step10_trainer_res152v2.py    --out_dir "10_ResNet152V2"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "yes"  --is_testing "yes" --is_extract "yes"  

#---------- EffNet
#python step10_trainer_effb7.py    --out_dir "10_Effb7"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "yes"  --is_testing "yes" --is_extract "yes"  
#python step10_trainer_effv2l.py    --out_dir "10_EffV2L"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "yes"  --is_testing "yes" --is_extract "yes"  

#---------- DenseNet
#python step10_trainer_den201.py    --out_dir "10_Den201"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "yes"  --is_testing "yes" --is_extract "yes"  

#---------- NastNet
#python step10_trainer_nasnet.py    --out_dir "10_NasNet"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "yes"  --is_testing "yes" --is_extract "yes"  

#---------- ConvTiny
#python step10_trainer_convtiny.py    --out_dir "10_ConvTiny"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "yes"  --is_testing "yes" --is_extract "yes"  
#python step10_trainer_convlarge.py    --out_dir "10_ConvLarge"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "yes"  --is_testing "yes" --is_extract "yes"  

#--------Mobile
#python step10_trainer_mobv2.py    --out_dir "10_MobV2"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "yes"  --is_testing "yes" --is_extract "yes"  
#python step10_trainer_mobv1.py    --out_dir "10_MobV1"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "yes"  --is_testing "yes" --is_extract "yes"  

#---------- InceptionV3
#python step10_trainer_incV3.py    --out_dir "10_InceptionV3"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "yes"  --is_testing "yes" --is_extract "yes"  

#python step10_trainer_incResV2.py  --out_dir "10_IncResV2"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "yes"  --is_testing "yes" --is_extract "yes"  

#------------------ Extract
#python step11_extract.py  --out_dir "10_Effb7"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "no"  --is_testing "no" --is_extract "yes"  
#
#python step11_extract.py  --out_dir "10_ResNet50"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "no"  --is_testing "no" --is_extract "yes"  
#python step11_extract.py  --out_dir "10_ResNet152V2"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "no"  --is_testing "no" --is_extract "yes"  
#
#python step11_extract.py  --out_dir "10_IncResV2"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "no"  --is_testing "no" --is_extract "yes"  
#python step11_extract.py  --out_dir "10_InceptionV3"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "no"  --is_testing "no" --is_extract "yes"  
#
#python step11_extract.py  --out_dir "10_MobV2"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "no"  --is_testing "no" --is_extract "yes"  
#python step11_extract.py  --out_dir "10_MobV1"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "no"  --is_testing "no" --is_extract "yes"  
#
#python step11_extract.py  --out_dir "10_ConvTiny"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "no"  --is_testing "no" --is_extract "yes"  
#python step11_extract.py  --out_dir "10_ConvLarge"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "no"  --is_testing "no" --is_extract "yes"  
#
#python step11_extract.py  --out_dir "10_Den201"    --train_dir   "./../01_split/fold_02_rotate/train" --test_dir "./../01_split/fold_01/test" --is_training "no"  --is_testing "no" --is_extract "yes"  


##---------------- Compute FLOP
#python step12_report_flop.py  --h5_dir  "./10_MobV1/model/best_model.h5"
#python step12_report_flop.py  --h5_dir  "./10_MobV2/model/best_model.h5"
#
#python step12_report_flop.py  --h5_dir  "./10_ResNet50/model/best_model.h5"
#python step12_report_flop.py  --h5_dir  "./10_ResNet152V2/model/best_model.h5"
#
#python step12_report_flop.py  --h5_dir  "./10_IncResV2/model/best_model.h5"
#python step12_report_flop.py  --h5_dir  "./10_InceptionV3/model/best_model.h5"
#
#python step12_report_flop.py  --h5_dir  "./10_Den201/model/best_model.h5"
#
#python step12_report_flop.py  --h5_dir  "./10_Effb7/model/best_model.h5"
#
#python step12_report_flop.py  --h5_dir  "./10_ConvTiny/model/best_model.h5"
#python step12_report_flop.py  --h5_dir  "./10_ConvLarge/model/best_model.h5"



