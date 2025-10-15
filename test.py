import numpy as np


intri = np.load('/data1/home/xu_zifan/ReCamMaster/example_test_data_plucker/intrinsics/1.npy')
print(intri.shape)
print(intri)
# intri[:, 0] /= intri[:, 2] * 2  # Normalize focal length x
# intri[:, 1] /= intri[:, 3] * 2  # Normalize focal length y
# intri[:, 2] = 0.5  # Set principal point x to center
# intri[:, 3] = 0.5  # Set principal point y to center
# print(intri[0])
# print(intri.shape)  # (80, 4)
# np.save('./example_test_data_plucker/intrinsics/1_normalized.npy', intri)