import os

def get_soma_coordinates(swc_file):
    with open(swc_file, 'r') as f:
        for line in f:
            if line.startswith('#'):  
                continue
            parts = line.split()
            if len(parts) > 1 and int(parts[1]) == 1: 
                soma_coordinates = (float(parts[2]), float(parts[3]), float(parts[4])) 
                return soma_coordinates
    return None

def get_all_soma_coordinates(directory):
    soma_coords = []
    for filename in os.listdir(directory):
        if filename.endswith(".swc"):
            filepath = os.path.join(directory, filename)
            coords = get_soma_coordinates(filepath)
            if coords:  
                filename_without_extension = os.path.splitext(filename)[0] 
                soma_coords.append((filename_without_extension, *coords))
    return soma_coords

def save_soma_coordinates_to_txt(soma_coords, output_file):
    with open(output_file, 'w') as f:
        for filename, x, y, z in soma_coords:
            f.write(f"{filename} {x} {y} {z}\n")

swc_directory = "/home/ylx/fly/data_rename"  
output_file = "./data/soma_coordinates.txt"  

soma_coordinates = get_all_soma_coordinates(swc_directory)

save_soma_coordinates_to_txt(soma_coordinates, output_file)

print(f"Soma coordinates saved to {output_file}")
