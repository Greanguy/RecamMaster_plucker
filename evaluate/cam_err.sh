# python ./cam_err/cam_err.py \
#   --gt ./cam_err/c2w_cam01_trans.json \
#   --est ./vipe/cam01_v0.json \
#   --key cam09 \
#   --out ./cam_err/cam09.csv \
#   --convention cam2world   # or world2cam, vipe提取的视频外参是c2w矩阵, 同时需要保证ground true的外参也是c2w矩阵并且坐标轴保持一致

python ./cam_err/cam_err.py \
  --gt /data1/home/liu_kai/RecamMaster_plucker/evaluate/cam_err/c2w_cam09_trans.json \
  --est_dir /data1/home/liu_kai/RecamMaster_plucker/evaluate/cam_err/cam09 \
  --key cam09 \
  --convention cam2world \
  --csv_out /data1/home/liu_kai/RecamMaster_plucker/evaluate/cam_err/cam09_all.csv \
  --ignore_first

# 注意坐标轴序列和方向就行了,做cam_err估计的时候
# vipe估计出来的pose是c2w矩阵,旋转矩阵是单位阵的格式:[1,0,0][0,1,0][0,0,1] 位移向量应该都是[tx,ty,tz]应该不需要管,一般处理的时候不会调换它们的位置
# recammaster原始相机轨迹也是c2w矩阵(做了个转置)