from PIL import Image
import numpy as np

def update_grid(map_x, map_y, grid, coord_x=0, coord_y=0):
    '''
    update_grid takes a grid (occupancy map) and updates it.

    map_x - shape of the map in the x coordinate
    map_y - shape of the map in the y coordinate

    grid - narray type grid (can be as big as you want, but no bigger than the shape of the map itself)
    should be initialised with zeros for all elements

    coord_x - current x coordinate
    coord_y - current y coordinate
    '''
    idx_x = np.arange(map_x)
    idx_y = np.arange(map_y)
    idx_x = np.array_split(idx_x, grid.shape[0])
    idx_y = np.array_split(idx_y, grid.shape[1])
    change_x = None
    change_y = None
    for i in range(np.shape(idx_x)[0]):
        if coord_x in idx_x[i]:
            change_x = i
    for i in range(np.shape(idx_y)[0]):
        if coord_y in idx_y[i]:
            change_y = i
    grid[change_x, change_y] = 1
    return grid

im = np.array(Image.open('map_images/my_way_home.wad_MAP01.png').convert('L')).shape
a = update_grid(map_x=im[0], map_y=im[1], grid = np.zeros((8,8)), coord_x=42, coord_y=102)
print(a)