from PIL import Image
import json
image = Image.new('RGB', (300, 8))

pixles = image.load()


map = json.load(open("pixel_map.json"))
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ!?"
for i,l in enumerate(letters):
    for x,y in map[l]:
        pixles[x+i*6,y] = (255,255,255)

image.save("arg.png")
