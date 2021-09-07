import argparse
from matplotlib.pyplot import imread, imsave

from tileset import Tileset
from wfc import Wave

if __name__=="__main__":
    # Arguments definition
    parser = argparse.ArgumentParser(
        description="Apply the WafeFunctionCollapse algorithm.")
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--tile_size", "-t", action="store", type=int, default=4)
    parser.add_argument("--output_size", "-s", action="store", nargs=2, type=int, default=[10,10])
    parser.add_argument("--wrap_horizontal", "-wh", action="store_true", default=False)
    parser.add_argument("--wrap_vertical", "-wv", action="store_true", default=False)

    # Argument parsing :)
    args = parser.parse_args()

    # Read input image
    img_in = imread(args.input_file)

    # Generate tileset
    tiles = Tileset()
    tiles.process_image(img_in, tile_size=args.tile_size,
        wrap_horizontal=args.wrap_horizontal,
        wrap_vertical=args.wrap_vertical)

    # Generate image
    wave = Wave()
    wave.generate(tiles, args.output_size[0], args.output_size[1])
    img_out = tiles.generate_image(wave.get_output())

    # Output final image
    imsave(args.output_file, img_out)
