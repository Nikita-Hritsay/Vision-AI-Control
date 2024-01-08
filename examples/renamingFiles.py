import os

def rename_files(folder_path, prefix):
    files = os.listdir(folder_path)
    i = 0
    for file_name in files:
        new_name = f"{prefix} {file_name}.png"

        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} to {new_name}")

folder_path = '../eyeTrainDataset/up'

prefix = 'down '

rename_files(folder_path, prefix)
