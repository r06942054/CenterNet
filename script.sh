

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

cd /home/omnieyes/renjie/GitHub/CenterNet/src/

# res_18
python main.py ctdet --exp_id omnieyes_res_18 --batch_size 32 --master_batch 1 --lr 1.25e-4  --gpus 0 --arch res_18 --head_conv 64 --num_epochs 200 --input_h 480 --input_w 800 --flip 0 --no_color_aug
python test.py --exp_id omnieyes_res_18 --not_prefetch_test ctdet --load_model /home/omnieyes/renjie/GitHub/CenterNet/exp/ctdet/omnieyes_res_18/model_best.pth --arch res_18 --head_conv 64 --input_h 480 --input_w 800

# dla_34
# python main.py ctdet --exp_id omnieyes_dla --batch_size 16 --master_batch 1 --lr 1.25e-4  --gpus 0 --num_epochs 200
# python test.py --exp_id omnieyes_dla --not_prefetch_test ctdet --load_model /home/omnieyes/renjie/GitHub/CenterNet/exp/ctdet/omnieyes_dla/model_best.pth

# mobilenetv3
python main.py ctdet --exp_id omnieyes_mobilenetv3 --batch_size 8 --master_batch 1 --lr 1.25e-4  --gpus 0 --arch mobilenetv3 --head_conv 64 --num_epochs 200 --input_h 480 --input_w 800 --flip 0 --no_color_aug
python test.py --exp_id omnieyes_mobilenetv3 --not_prefetch_test ctdet --load_model /home/omnieyes/renjie/GitHub/CenterNet/exp/ctdet/omnieyes_mobilenetv3/model_best.pth --arch mobilenetv3 --head_conv 64 --input_h 480 --input_w 800 --flip 0 --no_color_aug

# mobilenetv3_2
python main.py ctdet --exp_id omnieyes_mobilenetv3_2 --batch_size 16 --master_batch 1 --lr 1.25e-4  --gpus 0 --arch mobilenetv3 --head_conv 64 --num_epochs 200 --input_h 480 --input_w 800 --flip 0 --no_color_aug
python test.py --exp_id omnieyes_mobilenetv3_2 --not_prefetch_test ctdet --load_model /home/omnieyes/renjie/GitHub/CenterNet/exp/ctdet/omnieyes_mobilenetv3_2/model_best.pth --arch mobilenetv3 --head_conv 64 --input_h 480 --input_w 800 --flip 0 --no_color_aug

# run /home/omnieyes/renjie/GitHub/CenterNet/src/result_to_tfrecord.ipynb

# export PYTHONPATH=$PYTHONPATH:/home/omnieyes/renjie/TF_models/research:/home/omnieyes/renjie/TF_models/research/slim

# cd /home/omnieyes/renjie/OmniEyes_forOD/OmniEyes
# git checkout master

# python data_post_processing/inspect_record.py /home/omnieyes/renjie/GitHub/CenterNet/exp/ctdet/omnieyes_res_18/ \
# -s 0.5 \
# -i 0.5 \
# -o /home/omnieyes/renjie/GitHub/CenterNet/exp/ctdet/omnieyes_res_18/

# python data_post_processing/inspect_record.py /home/omnieyes/renjie/GitHub/CenterNet/exp/ctdet/omnieyes_dla/ \
# -s 0.5 \
# -i 0.5 \
# -o /home/omnieyes/renjie/GitHub/CenterNet/exp/ctdet/omnieyes_dla/

# python data_post_processing/inspect_record.py /home/omnieyes/renjie/GitHub/CenterNet/exp/ctdet/omnieyes_mobilenetv3/ \
# -s 0.5 \
# -i 0.5 \
# -o /home/omnieyes/renjie/GitHub/CenterNet/exp/ctdet/omnieyes_mobilenetv3/


# to onnx (輸出須改為list才能轉成onnx，https://github.com/xingyizhou/CenterNet/issues/77#issuecomment-503443329，但改了之後無法train dla，能train res18)
# run /home/omnieyes/renjie/GitHub/CenterNet/src/CenterNet_to_ONNX.ipynb









### Data to X299 ###

#安裝必須是conda install pytorch=0.4.1 cuda92 torchvision -c pytorch

# on 550
rsync -avrh /home/omnieyes/renjie/GitHub/CenterNet/data omnieyes@211.23.144.72:/home/omnieyes/renjie/GitHub/CenterNet

# on 211.23.144.72
rsync -avrh /home/omnieyes/renjie/GitHub/CenterNet/data omnieyes@192.168.0.24:/home/omnieyes/renjie/GitHub/CenterNet