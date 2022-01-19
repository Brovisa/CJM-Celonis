# Initial data wrangling + visualization
library(bupaR)
library(dplyr)
library(daqapo)
library(stringr)

source("data-retrieval.R")
activities = retrieve_app_cjm()

# Our data needs to have
# a timestamp -> activityTime
# a case identifier -> id
# an activity label -> activity
# a activity instance identifier -> make unique
# a transactional life cycle stage -> 'completed' everywhere
# a resource identifier -> '0' (but maybe use type of device/session)

activities = activities %>% 
  mutate(
    lifecycle = as.factor("completed"), 
    instance_id = 1:nrow(activities),
    resource = NA,
    activityTime = lubridate::as_datetime(
      activityTime, tz = "Europe/Amsterdam"
    ),
    # this is domain knowledge about the website: we consider each member visited a single type of activity
    activity = if_else(str_detect(pagePath, "^\\/\\w+(-[a-z]+)+"), "PAGEVIEW - /some-team-member", activity),
    activity = if_else(str_detect(pagePath, "^\\/articles\\/"), "PAGEVIEW - /article/some-article", activity),
    activity = if_else(str_detect(pagePath, "^\\/\\?"), "PAGEVIEW - /", activity),
    activity = if_else(str_detect(pagePath, "^\\/translate_c"), "PAGEVIEW - translate", activity)
  )
act_event = activities %>% 
  eventlog(
    case_id = "id",
    activity_id = "activity",
    activity_instance_id = "instance_id",
    lifecycle_id = "lifecycle",
    timestamp = "activityTime",
    resource_id = "resource"
  )

# act_event_agg = act_event$activity[str_detect(act_event$landingPagePath, "^\\/\\w+(-[a-z]+)+"),] = as.factor("PAGEVIEW /some-team-member")

# Some visualization
act_event %>% process_map()
act_event %>% filter_activity_frequency(percentage = 0.99) %>% process_map()
act_event %>% processmonitR::activity_dashboard()

act_event %>% filter_activity_frequency(percentage = 0.99) %>% processanimateR::animate_process(mode = "relative", epsilon = 1)


