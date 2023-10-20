import os

# Specify the folder path
folder_path = './img_out'

# Define a list of common image file extensions
common_image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']

# Check if the folder exists
if os.path.exists(folder_path):
    # List all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate through the files and remove common image files
    for filename in file_list:
        if any(filename.lower().endswith(ext) for ext in common_image_extensions):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f'Removed: {file_path}')
            except OSError as e:
                print(f'Error deleting {file_path}: {e}')
    print('Cleanup complete.')
else:
    print(f'Folder {folder_path} does not exist.')