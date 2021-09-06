from matplotlib.pyplot import imread, imsave

from tileset import Tileset
from wfc import Wave

if __name__=="__main__":
    # Read input image
    img = imread("img/Dungeon.png")
    # Generate tileset
    tiles = Tileset()
    tiles.process_image(img, tile_size=4, wrap_horizontal=True, wrap_vertical=True)
    # Generate image
    wave = Wave()
    wave.generate(tiles, 25, 25)
    # Output final image
    imsave("output_sample.png", wave.get_output())
