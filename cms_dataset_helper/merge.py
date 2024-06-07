import os
import shutil

# goal: 将/home/kms/wangyu_dtst/single_data和/home/kms/wangyu_dtst/single_data_200k/4k_single_chromo中的内容合并到一起。
# 这两个文件夹都是单根染色体数据集的根目录。

def merge(path1='/home/kms/wangyu_dtst/single_data',
          path2='/home/kms/wangyu_dtst/single_data_200k/single_data_200k'):
    # 将path1中的图像拷贝到path2

    # 统计path2图像个数：
    image_count_in_path1 = 0
    for i in range(24):
        sub_path1 = os.path.join(path1, str(i))
        image_count_in_path1 += len(os.listdir(sub_path1))

    # 将path1的图像copy到path2
    print(image_count_in_path1)
    finished = 0
    for i in range(24):
        sub_path1 = os.path.join(path1, str(i))
        sub_path2 = os.path.join(path2, str(i))
        image_names = os.listdir(sub_path1)
        for img_name in image_names:
            shutil.copy(src=os.path.join(sub_path1, img_name),
                        dst=os.path.join(sub_path2, img_name))
            finished += 1

            if finished % 100 == 0:
                print(f'\r{finished} / {image_count_in_path1} finished..')



if __name__ == '__main__':
    merge()


