import numpy as np
import heapq

from pdb import set_trace as bp

# TODO:
# - [collapse] at the edges, use the full tile

class CollapseContradiction(Exception):
    """ Raised when a pixel has no possible state to collapse into. """
    def __init__(self, x, y):
        Exception.__init__(self)
        self._x = x
        self._y = y
    def __str__(self):
        return 'Contradiction at {},{}'.format(self._x,self._y)

class Wave():
    def __init__(self):
        self.output = None
        self.bytemap = None
        self.collapsed = None
        self.tiles = None
    def run(self, t, n, m):
        """ Generates an output of size (n,m) from a tileset t. """
        self.tileset = t
        self.n = n
        self.m = m
        tile_size = t.tiles[0].shape[0]
        self.shift_h = 0 if t.wrap_horizontal else tile_size
        self.shift_v = 0 if t.wrap_vertical else tile_size

        # Initialize the output and the bitmap
        self.output = np.empty( (n,m), dtype=int )
        self.bytemap = np.ones( (n-self.shift_h,m-self.shift_v,t.num_tiles), dtype=bool )
        self.collapsed = np.zeros( (n-self.shift_h,m-self.shift_v), dtype=bool )

        # select/observe/propagate loop
        while not np.all(self.collapsed):
            # Select a pixel whose byte has lowest entropy
            i,j = self.select()
            # Observe the pixel. Propagation is implicit
            self.observe(i,j)
    def select(self):
        """ Returns a place to collapse. """
        # Note: all tiles have equal probabilities
        possibilities = np.sum(self.bytemap, axis=-1)
        entropies = np.log(possibilities)
        entropies[self.collapsed] = np.inf
        i,j = np.unravel_index(np.argmin(entropies), entropies.shape)
        return i,j
    def observe(self, i, j):
        """ Collapse a pixel by selecting a possible value. """
        # Choose a tile among the possible ones
        tile_num = np.random.choice(np.arange(self.tileset.num_tiles)[self.bytemap[i,j]])
        # Collapse the tile
        self.collapse(i, j, tile_num)
        # Propagate the infor
        self.propagate(i,j)
    def propagate(self, i, j):
        """ Update the neighboring tiles with the constraints formed by a cell
            state. """
        self.update((i-1) % self.n-self.shift_h, j % self.m-self.shift_v, self.bytemap[i,j] @ self.tileset.constraints_h.T)
        self.update((i+1) % self.n-self.shift_h, j % self.m-self.shift_v, self.bytemap[i,j] @ self.tileset.constraints_h)
        self.update(i % self.n-self.shift_h, (j-1) % self.m-self.shift_v, self.bytemap[i,j] @ self.tileset.constraints_v.T)
        self.update(i % self.n-self.shift_h, (j+1) % self.m-self.shift_v, self.bytemap[i,j] @ self.tileset.constraints_v)
    def collapse(self, i, j, tile_idx):
        # Update the bytemap
        self.bytemap[i,j] = False
        self.bytemap[i,j,tile_idx] = True
        # Update the output
        self.output[i,j] = tile_idx
        # Mark the pixel as collapsed
        self.collapsed[i,j] = True
    def update(self, i, j, constraints):
        """ Update the information at cell (i,j) with new contraints. """
        # Is this pixel already collapsed?
        if self.collapsed[i,j]:
            return None
        # Do the constraints bring new information
        if np.all(self.bytemap[i,j] <= constraints):
            return None
        # Update the tile
        self.bytemap[i,j] *= constraints
        # Did we run into a contradiction?
        if np.sum(self.bytemap[i,j]) == 0:
            raise CollapseContradiction(i,j)
        # Did we left only 1 pixel left?
        if np.sum(self.bytemap[i,j]) == 1:
            self.collapse(i,j,np.nonzero(self.bytemap[i,j])[0][0])
        # Propagate the information around.
        self.propagate(i,j)
    def get_output(self):
        return self.output
