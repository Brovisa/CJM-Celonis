# Google Analytics Mining
library(readr)
library(dplyr)
library(lubridate)
library(bupaR)
library(stringr)

# Read data
ga_min = read_csv(
    "../data/raw/google_analytics_sample_minimum.csv",
    col_types = cols(fullVisitorId = col_character())
)
ga_min$type <- as.factor(ga_min$type)
ga_min$eventAction <- as.factor(ga_min$eventAction)

# Filter quickview, product clicks (because these also have a related page view)
ga_min.filtered = ga_min %>%
    filter(!grepl("/quickview", pagePath, fixed = T)) %>%
    filter(!grepl("Product Click", eventAction))

# Create activities and set-up event log
ga_min.filtered = ga_min.filtered %>% mutate(
    activity = if_else(
        type == "PAGE",
        paste(toupper(type), pagePath, sep = " - "),
        paste(toupper(eventAction), eventLabel, sep = " - ")
    ),
    activity = as.factor(str_remove_all(activity, "[\\'\\+\\\\]")),
    type = NULL,
    eventAction = NULL,
    eventLabel = NULL,
    pagePath = NULL,
    lifecycle = as.factor("completed"),
    instance_id = 1:nrow(.),
    resource = NA,
    hitTime = lubridate::as_datetime(hitTime, tz = "Europe/Amsterdam")
)

# Create event log (sample)
ga_min.eventlog = ga_min.filtered %>% slice_head(n = 100000) %>%
    eventlog(
        case_id = "fullVisitorId",
        activity_id = "activity",
        activity_instance_id = "instance_id",
        lifecycle_id = "lifecycle",
        timestamp = "hitTime",
        resource_id = "resource"
    )

# Initial plotting (takes a while!)
ga_min.eventlog %>% filter_activity_frequency(percentage = 0.5) %>% process_map()
ga_min.eventlog %>% filter_activity_frequency(percentage = 0.5) %>% precedence_matrix(type = "relative") %>% plot()
ga_min.eventlog %>% trace_explorer(type = "infrequent")

## Setup similarity table for clustering

# We ignore sequences that are shorter than 30 pages, because otherwise the clustering becomes pretty random
# Another idea -> filter the homepage
ga_min.filtered.seq = ga_min.filtered %>% group_by(fullVisitorId) %>%
    summarise(seq = list(as.numeric(activity))) %>% filter(lengths((.$seq)) > 30)

distance <- function(a, b) {
    length(base::intersect(a, b)) / length(base::union(a, b))
}

# This takes a while
n = 4000
a = vector('list', length = n)
mydist = matrix(NA, n, n)
set.seed(1)
ga_min.sample = ga_min.filtered.seq %>% slice_sample(n = n) %>% pull()
for (i in 1:n) {
    # a[[i]] = vector('list', length = n - i + 1)
    for (j in i:n) {
        # a[[i]][[j]] = distance(pull(ga_min.filtered.seq[i,2]), pull(ga_min.filtered.seq[j,2]))
        mydist[i, j] = distance(ga_min.sample[[i]], ga_min.sample[[j]])
        mydist[j, i] = mydist[i, j]
    }
}
mydist.d = as.dist(mydist)
fit = hclust(d = mydist.d)
plot(fit, labels = F)
grps = cutree(fit, k = 10)
rect.hclust(fit, k = 12, border = "red")

## Another form of clustering: bag of activities
num = 100
ga_min.bagofactiv = (
    ga_min.filtered %>%
        group_by(fullVisitorId) %>%
        slice_sample(n = num) %>%
        group_by(fullVisitorId, activity) %>%
        add_tally() %>%
        ungroup() %>%
        tidyr::pivot_wider(
            names_from = activity,
            values_from = n,
            values_fill = 0
        ) %>%
        group_by(fullVisitorId) %>%
        summarise(across(c(
            which(colnames(.) == "resource"):ncol(.) - 1
        ), sum))
)

tmp = ga_min.filtered %>%
    group_by(fullVisitorId, activity) %>%
    add_tally() %>%
    ungroup()

tmp2 = tmp %>% slice_head(n = 10000) %>%
    tidyr::pivot_wider(names_from = activity,
                       values_from = n,
                       values_fill = 0) %>%
    group_by(fullVisitorId) %>%
    summarise(across(c((
        which(colnames(.) == "resource") - 1
    ):(ncol(
        .
    ) - 1)), sum))

pca_res = prcomp(tmp2 %>% select(-c(1, 2)), center = T, scale = T)
summary(pca_res)
# perform clustering
clusters = kmeans(pca_res$x, centers = 10)
# plot results from PCA
library(factoextra)
fviz_pca_ind(
    pca_res,
    col.ind = "cos2",
    # Color by the quality of representation,
    geom = c("point"),
    gradient.cols = "npg",
)
# plot clusters
# palette("ggplot2")
palette(rainbow(10))
plot(
    pca_res$x[, 4],
    pca_res$x[, 3],
    col = clusters$cluster,
    cex = 1,
    pch = 19
)

## Fuzzy k-medoids
library(fclust)
# FKM.med(tmp2[-1:-2], k = 10)
res = Fclust(mydist, distance = T)
plot(res)
res2 = NEFRC(
    mydist,
    k = 10,
    m = 2,
    RS = 1,
    maxit = 1e6
)
