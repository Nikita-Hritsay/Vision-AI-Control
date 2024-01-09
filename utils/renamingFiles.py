import os

def rename_files(folder_path):
    files = os.listdir(folder_path)
    i_up = 1
    i_down = 1
    i_right = 1
    i_left = 1
    for file_name in files:
        direction = file_name.split(' ')[0]
        if direction == "up":
            new_name = f"{direction} {i_up}.png"
            i_up += 1
        if direction == "down":
            new_name = f"{direction} {i_down}.png"
            i_down += 1
        if direction == "left":
            new_name = f"{direction} {i_left}.png"
            i_left += 1
        if direction == "right":
            new_name = f"{direction} {i_right}.png"
            i_right += 1

        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} to {new_name}")

folder_path = '../eyeTrainDataset'

rename_files(folder_path)
