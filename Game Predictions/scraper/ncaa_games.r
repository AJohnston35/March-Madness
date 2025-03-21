# Load necessary packages
library(tictoc)
library(progressr)
library(hoopR)

# Loop through each year from 2002 to 2025
for (year in 2025:2025) {
  # Start the timer for each year
  tictoc::tic(paste("Year:", year))

  # Load the basketball data with progress tracking
  progressr::with_progress({
    # Load the team box data for the current season (for that specific year)
    mbb_team_box <- hoopR::load_mbb_team_box(year)
  })

  # Stop the timer
  tictoc::toc()

  # Check if the data was loaded correctly
  if (exists("mbb_team_box")) {
    # Save the data to a CSV file for the specific year
    file_path <- paste0("player_games_", year, ".csv")
    write.csv(mbb_team_box, file_path, row.names = FALSE)
    cat(paste("Data for year", year, "saved to", file_path, "\n"))
  } else {
    cat(paste("Error: Data for year", year, "was not loaded.\n"))
  }
}
