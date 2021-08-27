import numpy as np

class Tileset():
    def __init__():
        self.num_tiles = 0
        self.contraints_h = set({}) # Set. (i,j) present iff i can be on top of j.
        self.contraints_v = set({}) # (i,j) present iff i can be at the left of j.
        self.wrap_horizontal = False
        self.wrap_vertical = False
        self.tiles = [] # List of images. Implicite tile_num <--> image mapping.
    def process_image(img, tile_size=2,
                      wrap_horizontal=False, wrap_vertical=False):
        """ Takes as input an image, slice it into tiles and compute
            constraints."""
        self.wrap_horizontal = wrap_horizontal
        self.wrap_vertical = wrap_vertical
        n,m = img.shape
        shift_h = 0 if wrap_horizontal else tile_size
        shift_v = 0 if wrap_vertical else tile_size

        # Image, as a set of tiles
        a = np.arange((n-shift_h)*(m-shift_v)).reshape( ((n-shift_h),(m-shift_v)) )

        # Step 1: create all tiles
        for i in range(n-shift_h):
            # Extract the horizontal part
            h_slice = img.take(np.arange(i,i+tile_size), axis=0, mode="wrap")
            for j in range(m-shift_v):
                # Finish extracting tile
                new_tile = h_slice.take(np.arange(j,j+tile_size), axis=1, mode="wrap")
                # Check if tile already exists
                for k,old_tile in enumerate(self.tiles):
                    if np.equal(new_tile, old_tile):
                        # Tile already known
                        a[i,j] = k
                        break
                else:
                    # Tile unknown
                    self.tiles.append(new_tile)

        # Step 2: compute adjacency contraints
        # (Note: this could be made fast with existing numpy functions.)
        for i in range(n-shift_h):
            for j in range(m-shift_v):
                self.contraints_h.add( (a[i,j],a[i+1 % n,j]) )
                self.contraints_v.add( (a[i,j],a[i,j+1 * m]) )
