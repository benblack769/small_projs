from PIL import Image
import numpy as np
import json
import random

def generate_letter_map(map, letters):
    coordinates = []
    for i,l in enumerate(letters.upper()):
        for x,y in map[l]:
            coordinates.append((x+i*6, y))
    return coordinates

def to_6d_coords(coords):
    random.shuffle(coords)
    arr = np.array(coords)
    coords = arr
    matrix = np.random.normal(size=(2,6))
    result = coords @ matrix
    rand_result = result + np.random.normal(size=result.shape)*0.02
    return rand_result

def create_clue(map, letter, name):
    coords = generate_letter_map(map, letter)
    mixed = to_6d_coords(coords)
    tsv_result = "\n".join("\t".join(str(val) for val in vals) for vals in mixed)
    print(tsv_result,file=open(name,'w'))

if __name__ == "__main__":
    map = json.load(open("pixel_map.json"))
    create_clue(map, "darwin", "clue1.tsv")
    create_clue(map, "sewing", "clue2.tsv")
    create_clue(map, "barn", "clue3.tsv")
