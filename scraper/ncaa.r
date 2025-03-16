# Load necessary packages
library(tictoc)
library(progressr)
library(hoopR)

# Loop through each year from 2002 to 2025
for (year in 2003:2025) {
  
  # Start the timer for each year
  tictoc::tic(paste("Year:", year))
  
  # Load the basketball player data with progress tracking
  progressr::with_progress({
    # Load the player box data for the current season (for that specific year)
    mbb_player_box <- hoopR::load_mbb_player_box(year)
  })
  
  # Stop the timer
  tictoc::toc()
  
  # Check if the data was loaded correctly
  if (exists("mbb_player_box")) {
    # Save the data to a CSV file for the specific year
    file_path <- paste0("player_games_", year, ".csv")
    write.csv(mbb_player_box, file_path, row.names = FALSE)
    cat(paste("Player data for year", year, "saved to", file_path, "\n"))
  } else {
    cat(paste("Error: Player data for year", year, "was not loaded.\n"))
  }
}
