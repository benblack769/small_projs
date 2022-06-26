from PIL import Image
import json

def is_set(img, x, y):
    pix = max(img.getpixel((x, y))[:3])
    return pix < 200

def is_divisor(img, x, yl):
    return not any(is_set(img, x, y+yl) for y in range(5))

def get_divisors(img):
    divisors = []
    for yl in range(4):
        yldiv = []
        x = 0
        yldiv.append(0)
        while x < 39:
            while x < 39 and is_divisor(img, x, yl*6):
                x += 1

            while x < 39 and not is_divisor(img, x, yl*6):
                x += 1

            yldiv.append(x)
            x += 1
        yldiv.append(39)
        divisors.append(yldiv)
    return divisors

def read_img():
    img = Image.open("pixelart.PNG")
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ!?"
    divisors = get_divisors(img)
    pixmap = {}
    # print(divisors)
    for l in range(len(letters)):
        c = letters[l]
        xli = l % 8
        yl = l // 8
        xls, xle = divisors[yl][xli:xli+2]
        # print(xls)
        points = []
        for x in range(0,xle-xls):
            for y in range(5):
                if is_set(img, xls+x, yl*6+y):
                    points.append([x,y])
        pixmap[c] = points

    res = json.dumps(pixmap, sort_keys=True)
    print(res)

if __name__ == "__main__":
    read_img()
