library(ggplot2)
library(viridis)

# 1. plot barplot
ego_cc_df$BgRatio <- as.character(ego_cc_df$BgRatio) 
p <- ggplot(ego_cc_df, aes(x = reorder(Description, -pvalue), y = -log10(pvalue), fill = -log10(pvalue))) +
  geom_bar(stat = "identity", width = 0.7, color = "black", size = 0.2) +
  coord_flip() +
  scale_fill_viridis(option = "C", direction = -1, name = "-log10(p-value)") +
  labs(
    title = "Top 10 Enriched GO:CC Terms",
    x = NULL,
    y = "-log10(p-value)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "grey80", size = 0.2),
    axis.text = element_text(color = "black", size = 12),
    axis.title.y = element_text(size = 13),
    axis.title.x = element_text(size = 13),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
    legend.position = "right",
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 11)
  )
ggplot2::ggsave("GO_CC.tif", plot = p, width = 8, height = 6, units = "in", dpi = 500)

#####################
# 2. plot in cell
new_column_values <- c(
  "Extracellular region", "Extracellular region", "Extracellular region",
  "Cytoplasm", "lipid droplet", "Nucleus",
  "Plasma membrane", "Nucleus", "Plasma membrane", "Nucleus"
)

ego_cc_df$Subcellular_location[1:10] <- new_column_values

library(dplyr)

ego_cc_df_summary <- ego_cc_df %>%
  group_by(Subcellular_location) %>%
  summarise(Count = sum(Count, na.rm = TRUE)) %>%
  ungroup()

ego_cc_df_summary <- ego_cc_df_summary %>%
  mutate(Proportion = Count / sum(Count))

library(scales)

log_prop <- log10(ego_cc_df_summary$Proportion + 1e-6)

color_fun <- col_numeric(
  palette = c("#609377", "#D8B2B2"),
  domain = range(log_prop, na.rm = TRUE)
)

ego_cc_df_summary$Color <- color_fun(log_prop)

#############
# 3. color bar
library(ggplot2)
library(scales)

ego_cc_df_summary$log_prop <- log10(ego_cc_df_summary$Proportion + 1e-6)

color_fun <- col_numeric(
  palette = c("#609377", "#D8B2B2"),
  domain = range(ego_cc_df_summary$log_prop, na.rm = TRUE)
)

colorbar_df <- data.frame(
  log_prop = seq(min(ego_cc_df_summary$log_prop), max(ego_cc_df_summary$log_prop), length.out = 100)
)
colorbar_df$color <- color_fun(colorbar_df$log_prop)

ggplot(colorbar_df, aes(x = 1, y = log_prop, fill = color)) +
  geom_tile() +
  scale_fill_identity() +
  scale_y_continuous(
    breaks = log10(c(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1)),
    labels = c("1e-6","1e-5","1e-4","1e-3","1e-2","1e-1","1"),
    name = "Proportion (log10 scale)"
  ) +
  theme_minimal() +
  theme(
    axis.title.x=element_blank(),
    axis.text.x=element_blank(),
    axis.ticks.x=element_blank()
  )

