python process2.py -i ./camera_extrinsics.json -o cam01-10_trans.json
# 将recammaster提供的外参的旋转矩阵转换成单位阵的格式:[1,0,0][0,1,0][0,0,1]

python extract.py
# 从中提取出cam01的参数