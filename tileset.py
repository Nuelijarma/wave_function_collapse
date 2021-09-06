import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

# TODO:
# - add counter of tile type, to get a distribution

class Tileset():
    def __init__(self):
        self.num_tiles = 0
        self.contraints_h = None # Sparse array. A[i,j]==1 iff i can be on top of j.
        self.contraints_v = None # A[i,j]==1 iff i can be at the left of j.
        self.wrap_horizontal = False
        self.wrap_vertical = False
        self.tiles = [] # List of images. Implicite tile_num <--> image mapping.
    def process_image(self, img, tile_size=2,
                      wrap_horizontal=False, wrap_vertical=False):
        """ Takes as input an image, slice it into tiles and compute
            constraints."""
        self.num_tiles = 0
        self.wrap_horizontal = wrap_horizontal
        self.wrap_vertical = wrap_vertical
        n,m,d = img.shape
        shift_h = 0 if wrap_horizontal else tile_size
        shift_v = 0 if wrap_vertical else tile_size

        # Image, as a set of tiles
        # a = np.arange((n-shift_h)*(m-shift_v)).reshape( ((n-shift_h),(m-shift_v)) )
        a = np.zeros( (n-shift_h,m-shift_v) )

        # Step 1: create all tiles
        for i in range(n-shift_h):
            # Extract the horizontal part
            h_slice = img.take(np.arange(i,i+tile_size), axis=0, mode="wrap")
            for j in range(m-shift_v):
                # Finish extracting tile
                new_tile = h_slice.take(np.arange(j,j+tile_size), axis=1, mode="wrap")
                # Check if tile already exists
                for k,old_tile in enumerate(self.tiles):
                    if np.all(np.equal(new_tile, old_tile)):
                        # Tile already known
                        a[i,j] = k
                        break
                else:
                    # Tile unknown
                    self.tiles.append(new_tile)
                    a[i,j] = self.num_tiles
                    self.num_tiles += 1

        # Step 2: compute adjacency contraints
        # 2a/ Collect indexed i,j for constraints (i can be next to j)
        i_h, j_h, i_v, j_v, = [], [], [], []
        for i in range(n-shift_h):
            for j in range(m-shift_v):
                i_h.append(a[i,j])
                j_h.append(a[(i+1) % n,j])
                i_v.append(a[i,j])
                j_v.append(a[i,(j+1) % m])
        # 2b/ Store constraints in (sparse) COO matrix
        self.constraints_h = coo_matrix( (np.ones((len(i_h),),dtype=bool), (i_h,j_h)), shape=(self.num_tiles, self.num_tiles))
        self.constraints_v = coo_matrix( (np.ones((len(i_v),),dtype=bool), (i_v,j_v)), shape=(self.num_tiles, self.num_tiles))
        # 2c/ Store constraints in (sparse) CSR matrix
        self.constraints_h = csr_matrix(self.constraints_h)
        self.constraints_v = csr_matrix(self.constraints_v)
