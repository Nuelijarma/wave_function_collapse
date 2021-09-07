import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from pdb import set_trace as bp

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
    def generate_image(self, input):
        """ Generate an image from a matrix of tile numbers. """
        # Compute output image shape
        n,m = input.shape
        out_shape_x = n + (0 if self.wrap_horizontal else self.tiles[0].shape[0]-1)
        out_shape_y = m + (0 if self.wrap_vertical else self.tiles[0].shape[1]-1)
        out_shape = (out_shape_x,out_shape_y) + self.tiles[0][0,0].shape
        output = np.empty(out_shape, dtype=self.tiles[0].dtype)
        # Fill output
        # Notes:
        # - min(i,n) and min(j,m) allows selecting the correct tile at the right
        #   and bottom borders when not wrapping. When i (resp. j) becomes
        #   greater than n (resp. m), the indexing of input[] stops increasing.
        # - Similarly, max(0,i-out_shape_x) allows selecting the proper pixel
        #   when not wrapping. When i becomes greater than n, pixels others than
        #   the top=left corners start being used.
        for i in range(out_shape_x):
            for j in range(out_shape_y):
                output[i,j] = self.tiles[input[min(i,n),min(j,m)]][max(0,i-n),max(0,j-m)]
        return output
