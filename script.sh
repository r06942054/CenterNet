

# https://blog.csdn.net/weixin_41765699/article/details/100118353?fbclid=IwAR0kS1niQki_SBODhimtSNOQ8aqSBYtuY8Vm3m0uB87m80blS0nLoAM2QZg


# copy data from /home/omnieyes/Label  to /home/omnieyes/renjie/GitHub/CenterNet/data

cd data
mkdir omnieyes
cd omnieyes
mkdir training_videos
mkdir testing_videos

# cp -r /home/omnieyes/Label/data_analysis_20200217/training_videos/* training_videos/
# cp -r /home/omnieyes/Label/data_analysis_20200217/testing_videos/* testing_videos/

# cp -r /home/omnieyes/Label/data_analysis_20200312/training_videos/* training_videos/

# cp -r /home/omnieyes/Label/data_analysis_20200413/training_videos/* training_videos/

# cp /home/omnieyes/Label/labels.txt labels.txt

# video to images
# conda activate tf1.12

# python /home/omnieyes/renjie/OmniEyes/tools/video_to_image.py \
# -d training_videos/ \
# -o training_images \
# -a

# python /home/omnieyes/renjie/OmniEyes/tools/video_to_image.py \
# -d testing_videos/ \
# -o testing_images \
# -a

# run /home/omnieyes/renjie/OmniEyes_forTools/OmniEyes/tools/omnieyes_to_cocoformat.ipynb

# modify some code...

# run
unset PYTHONPATH

conda activate CenterNet

cd ../../src

python main.py ctdet --exp_id omnieyes_res_18 --batch_size 32 --master_batch 1 --lr 1.25e-4  --gpus 0 --arch res_18 --head_conv 64 --num_epochs 100

python main.py ctdet --exp_id omnieyes_dla --batch_size 16 --master_batch 1 --lr 1.25e-4  --gpus 0 --num_epochs 100

python test.py --exp_id omnieyes_res_18 --not_prefetch_test ctdet --load_model /home/omnieyes/renjie/GitHub/CenterNet/exp/ctdet/omnieyes_res_18/model_best.pth

python test.py --exp_id omnieyes_dla --not_prefetch_test ctdet --load_model /home/omnieyes/renjie/GitHub/CenterNet/exp/ctdet/omnieyes_dla/model_best.pth