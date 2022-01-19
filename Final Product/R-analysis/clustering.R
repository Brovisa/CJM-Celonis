# Clustering ideas (for now, just K-means with pca)
library(factoextra)
# Rewrite to activities per id
act_n_per_id = (activities %>% group_by(id, activity)
     %>% add_tally() 
     %>% ungroup() 
     %>% tidyr::pivot_wider(names_from = activity, values_from = n, values_fill = 0)
     %>% group_by(id) %>% summarise(across(c(which(colnames(activities) == "resource"):ncol(.)-1), sum)))

# pca for better plotting
pca_res = prcomp(act_n_per_id %>% select(-id), center = T, scale = T)
summary(pca_res)
# perform clustering
clusters = kmeans(pca_res$x, centers = 10)
# append cluster determination
act_n_per_id$cluster = clusters$cluster
# plot results from PCA
fviz_pca_ind(pca_res,
             col.ind = "cos2", # Color by the quality of representation,
             geom = c("point"),
             gradient.cols = "npg",
)
# plot clusters
# palette("ggplot2")
palette(rainbow(10))
plot(pca_res$x[,4], pca_res$x[,3], col = clusters$cluster, cex = 1, pch = 19)
