import os
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
import random

def horizontal_flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def vertical_flip_image(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def rotate_image(image):
    return image.rotate(90, expand=True)

def distort_image(image):
    width, height = image.size
    xshift = abs(random.uniform(-0.3, 0.3) * width)
    new_width = width + int(round(xshift))
    return image.transform((new_width, height), Image.AFFINE,
                           (1, random.uniform(-0.3, 0.3), -xshift if random.random() < 0.5 else 0,
                            0, 1, 0), Image.BICUBIC)

def scale_image(image, scale_factor=1.5):
    width, height = image.size
    new_size = (int(width * scale_factor), int(height * scale_factor))
    return image.resize(new_size, Image.BICUBIC)

def random_erase(image, erase_factor=0.3):
    width, height = image.size
    erase_size = (int(width * erase_factor), int(height * erase_factor))
    erase_position = (random.randint(0, width - erase_size[0]), random.randint(0, height - erase_size[1]))
    image_copy = image.copy()
    for i in range(erase_size[0]):
        for j in range(erase_size[1]):
            image_copy.putpixel((erase_position[0] + i, erase_position[1] + j), (0, 0, 0))
    return image_copy

def add_noise(image, mean=0, std=0.1):
    image_array = np.array(image) / 255.0
    noise = np.random.normal(mean, std, image_array.shape)
    noisy_image = np.clip(image_array + noise, 0, 1) * 255
    return Image.fromarray(noisy_image.astype(np.uint8))

def cutout(image, cut_size_ratio=0.2):
    width, height = image.size
    cut_width = int(width * cut_size_ratio)
    cut_height = int(height * cut_size_ratio)
    cut_x = random.randint(0, width - cut_width)
    cut_y = random.randint(0, height - cut_height)
    image_copy = image.copy()
    for i in range(cut_width):
        for j in range(cut_height):
            image_copy.putpixel((cut_x + i, cut_y + j), (0, 0, 0))
    return image_copy

def adjust_brightness(image, factor=1.5):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor=1.5):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_saturation(image, factor=1.5):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def adjust_hue(image, factor=0.1):
    image = np.array(image.convert('HSV'))
    image[:, :, 0] = (image[:, :, 0].astype(np.int32) + int(factor * 255)) % 256
    return Image.fromarray(image, 'HSV').convert('RGB')

def gaussian_blur(image, radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius))

def shuffle_channels(image):
    image = np.array(image)
    np.random.shuffle(image.T)
    return Image.fromarray(image)

def to_grayscale(image):
    return image.convert('L').convert('RGB')

def translate_image(image, max_translate=(50, 50)):
    x_translate = random.randint(-max_translate[0], max_translate[0])
    y_translate = random.randint(-max_translate[1], max_translate[1])
    return image.transform(image.size, Image.AFFINE, (1, 0, x_translate, 0, 1, y_translate))

def perspective_transform(image, distortion_scale=0.2):
    width, height = image.size
    xshift = abs(distortion_scale * width)
    yshift = abs(distortion_scale * height)
    new_width = width + int(round(xshift))
    new_height = height + int(round(yshift))
    return image.transform((new_width, new_height), Image.PERSPECTIVE,
                           (1 - distortion_scale, 0, xshift // 2,
                            0, 1 - distortion_scale, yshift // 2,
                            -xshift // 2, -yshift // 2))

def crop_image(image, crop_factor=0.8):
    width, height = image.size
    new_width = int(width * crop_factor)
    new_height = int(height * crop_factor)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    return image.crop((left, top, left + new_width, top + new_height))

def augment_images_in_directory(input_dir, output_dir, augment_fn):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, file_name)
            image = Image.open(image_path)
            augmented_image = augment_fn(image)
            output_path = os.path.join(output_dir, file_name)
            augmented_image.save(output_path)

input_directory = '/data/Shared_Data/VQAv2/train2014'
output_directory = '/data/Shared_Data/VQAv2_flip/train2014'

augment_images_in_directory(input_directory, output_directory, horizontal_flip_image)
