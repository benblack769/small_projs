from PIL import Image

import numpy as np

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ!"


def generate_letters(fname):
    img = Image.open(fname)
    img.load()
    new_size = (36, 36)
    small_img = img.resize(new_size,resample=Image.Resampling.BICUBIC)
    np_img = np.array(small_img).mean(axis=2).astype("uint8").flatten()
    np_img[np_img < 55] = 0
    np_img[np_img > 205] = 0
    np_img[np_img > 0] = 255
    reshaped_img = np_img.reshape(new_size)
    print(reshaped_img.shape)
    for y in range(0,36,12):
        for x in range(0,36,12):
            yield reshaped_img[y:y+12,x:x+12]


def generate_all_letters():
    yield from generate_letters("abc.jfif")
    yield from generate_letters("jkl.jfif")
    yield from generate_letters("stu.jfif")


def write_letters():
    with open("letters.bin",'wb') as file:  
        for letter in generate_all_letters():
            file.write(letter.tobytes())


if __name__ == "__main__":
    write_letters()
