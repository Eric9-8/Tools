# 上海工程技术大学
# 崔嘉亮
# 开发时间：2021/10/28 16:46
# python install pillow
from PIL import Image
import os


# 分割图片
def cut_image(image, count):
    width, height = image.size
    item_height = int(height / count)
    item_width = width
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, count):
        box = (0, i * item_height, item_width, (i + 1) * item_height)
        box_list.append(box)
    print(box_list)
    image_list = [image.crop(box) for box in box_list]
    return image_list


# 保存分割后的图片
def save_images(image_list, name):
    index = 1
    for image in image_list:
        image.save('D:/Pytorch/WGAN_GP/utils/损伤类-2/' + str(name) + '-' + str(index) + '.jpg', 'JPEG')
        index += 1


if __name__ == '__main__':
    file_path = "/WGAN_GP/dataset/损伤2"  # 要分割的图片地址
    # image = Image.open(file_path)  # 读取图片
    for filename in os.listdir(file_path):
        image = Image.open(file_path + '/' + filename)
        image_list = cut_image(image, 8)  # 分割图片，分割成16张
        save_images(image_list, filename)  # 保存分割后的图片
