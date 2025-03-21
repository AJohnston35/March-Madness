import os

# Path to the logos directory
logos_dir = "espn_bracketology/assets/logos"

# Loop through all files in the directory
for filename in os.listdir(logos_dir):
    # Create the new filename by ensuring it ends with .png
    new_filename = filename.replace("-", "_")
    # Get full file paths
    old_path = os.path.join(logos_dir, filename)
    new_path = os.path.join(logos_dir, new_filename)
    
    # Rename the file
    os.rename(old_path, new_path)
    print(f'Renamed: {filename} -> {new_filename}')

print("Renaming complete!")
