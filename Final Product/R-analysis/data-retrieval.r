# This file is responsible for loading the data
# Here, we use a pre-defined DB. Later, we would like to retrieve the data
# from Google Analytics or Adobe Adwords

library(DBI)

retrieve_app_cjm <- function() {
  if (!DBI::dbCanConnect(RSQLite::SQLite(), "../data/raw/cjm-app.db")) {
    stop("Cannot connect to DB, please verify settings!")
  }
  con = DBI::dbConnect(RSQLite::SQLite(), dbname="data/cjm-app.db")
  activities = DBI::dbGetQuery(con, "SELECT * FROM cjm ORDER BY activitytime DESC LIMIT 10000") # retrieve all data
  DBI::dbDisconnect(con)
  return(activities)
}

retrieve_google_analytics <- function() {
  library(readr)
  ga_min = read_csv("../data/raw/google_analytics_sample_minimum.csv", 
                    col_types = cols(fullVisitorId = col_character()))
  return(ga_min)
}
