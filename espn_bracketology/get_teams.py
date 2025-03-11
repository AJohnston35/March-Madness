import os

def save_team_names_to_txt(logo_directory, output_file):
    team_names = []

    # Iterate through each file in the logo directory
    for filename in os.listdir(logo_directory):
        if filename.endswith('.png'):  # Assuming logos are in PNG format
            # Remove the file extension and replace underscores with spaces
            team_name = filename[:-4].replace('_', ' ').title()
            team_names.append(team_name)

    # Write the team names to a text file
    with open(output_file, 'w') as f:
        for name in team_names:
            f.write(f"{name}\n")

# Call the function with the appropriate directory and output file
save_team_names_to_txt('espn_bracketology/assets/logos', 'espn_bracketology/team_names.txt')
