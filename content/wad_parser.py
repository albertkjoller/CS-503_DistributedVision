import numpy as np


def parse_coordinate_line(coordinate_line: str):
    return float(coordinate_line[:-1].split(" ")[-1])

def parse_wad_file(path: str):
    print(f"Parsing {path}\n")
    with open(path, "rb") as file:
        binary_data = file.read()
        raw_string = binary_data.decode("utf-8", errors="ignore")
        lines = raw_string.split("\n")

    line_index = 0
    vertices = []
    while line_index < len(lines):
        line = lines[line_index]
        if line.startswith("vertex"):
            vertex_x = parse_coordinate_line(lines[line_index + 2])
            vertex_y = parse_coordinate_line(lines[line_index + 3])
            vertices.append([vertex_x, vertex_y])
            print(f"  * Added vertex {(vertex_x, vertex_y)}")
        
        line_index += 1

    vertices = np.array(vertices)
    print("\n\nResults:")
    print(f"  (x_min, x_max) = ({np.min(vertices[:, 0])}, {np.max(vertices[:, 0])})")
    print(f"  (y_min, y_max) = ({np.min(vertices[:, 1])}, {np.max(vertices[:, 1])})")
    print()

if __name__ == "__main__":
    parse_wad_file("setting/my_way_home.wad")
