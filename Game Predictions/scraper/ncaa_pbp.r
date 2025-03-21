# Load necessary packages
library(tictoc)
library(progressr)
library(hoopR)
library(dplyr) # Add dplyr for data manipulation

# Loop through each year from 2006 to 2025
for (year in 2006:2025) {
    # Start the timer for each year
    tictoc::tic(paste("Year:", year))

    # Load the basketball play-by-play data with progress tracking
    progressr::with_progress({
        mbb_pbp <- hoopR::load_mbb_pbp(year)
    })

    # Stop the timer
    tictoc::toc()

    # Check if the data was loaded correctly
    if (exists("mbb_pbp")) {
        # Keep only the first row for each unique game_id
        mbb_pbp_filtered <- mbb_pbp %>%
            group_by(game_id) %>%
            slice(1) %>%
            ungroup()

        # Save the filtered data to a CSV file
        file_path <- paste0("Data/pbp/pbp_", year, ".csv")
        write.csv(mbb_pbp_filtered, file_path, row.names = FALSE)
        cat(paste("Filtered data for year", year, "saved to", file_path, "\n"))
    } else {
        cat(paste("Error: Data for year", year, "was not loaded.\n"))
    }
}
