categories1 <- c("M", "M", "M", "I", "M", "I", "TM", "TM", "M", "TM")
ego_df$self_define_category <- categories1

categories2 <- c("M", "M", "M", "M", "M", "M")
kegg_df$self_define_category <- categories2

###########################################################################
library(dplyr)
library(tidyr)

combined_df <- bind_rows(
  ego_df %>% select(geneID, self_define_category),
  kegg_df %>% select(geneID, self_define_category)
)

df_long <- combined_df %>%
  select(geneID, self_define_category) %>%
  separate_rows(geneID, sep = "/") %>%  
  rename(Gene = geneID)

gene_cat_count <- df_long %>%
  group_by(Gene, self_define_category) %>%
  summarise(count = n(), .groups = "drop")

gene_cat_matrix <- gene_cat_count %>%
  pivot_wider(names_from = self_define_category, values_from = count, values_fill = 0)

print(gene_cat_matrix)

############################################################################
target_genes <- c("ppk12", "ppk28", "CG30487", "p38a", "spz", "Ctr1A", "CCAP", "tim", "t", "Trhn",
                  "Tpc2", "Zip42C.1", "ple", "e", "obst-A", "Lcp65Af", "Ccp84Ad", "kkv", "Edg78E", "slif",
                  "CG4476", "CG13796", "CG13793", "mnd", "CAH16", "CAH6", "Gs2", "CG10352", "CG10962",
                  "Cyp18a1", "PIG-X", "PIG-H", "PIG-F", "PIG-C", "CG11951", "CG4306", "S-Lap7", "CG1492",
                  "GstD5", "Alp5", "CG12766", "Trhn", "Alp11", "ple", "Cht7", "Cht9", "Cht4", "kkv")

filtered_gene_cat_matrix <- gene_cat_matrix %>%
  filter(Gene %in% target_genes)

#print(filtered_gene_cat_matrix)

library(dplyr)

filtered_gene_cat_matrix_new <- filtered_gene_cat_matrix %>%
  left_join(gene_info, by = c("Gene" = "external_gene_name")) %>%
  mutate(Gene = ifelse(is.na(flybase_gene_id), Gene, flybase_gene_id)) %>%
  select(-flybase_gene_id)
filtered_gene_cat_matrix_new <- filtered_gene_cat_matrix_new %>%
  select(Gene, M, I, TM)

write.csv(filtered_gene_cat_matrix_new, file = "filtered_gene_cat_matrix.csv", row.names = TRUE)

