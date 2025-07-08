library(clusterProfiler)
library(org.Dm.eg.db)
library(readr)
library(biomaRt)
library(dplyr)
library(DOSE) 
library(GseaVis)

# Data Preprocess
genes <- read_lines("positive_genes.txt")  

ensembl <- useEnsembl(
  biomart = "ensembl",
  dataset = "dmelanogaster_gene_ensembl",
  mirror = "asia"  
)

gene_info <- getBM(
  attributes = c("external_gene_name", "flybase_gene_id", "gene_biotype", "external_synonym", "description"),
  filters = "flybase_gene_id",
  values = genes,  
  mart = ensembl
)

gene_info$original_id <- paste0(gene_info$flybase_gene_id)

write_csv(gene_info, "converted_genes.csv")

#############################################################################
# Go Analyse
converted <- read_csv("converted_genes.csv")
gene_list <- converted$external_gene_name

gene_df <- bitr(gene_list,
                fromType = "SYMBOL",
                toType = "ENTREZID",
                OrgDb = org.Dm.eg.db)

head(gene_df)

ego <- enrichGO(gene         = gene_df$ENTREZID,
                OrgDb        = org.Dm.eg.db,
                keyType      = "ENTREZID",
                ont          = "BP",
                pAdjustMethod = "BH",
                pvalueCutoff  = 0.05,
                readable      = TRUE)

ego_cc <- enrichGO(gene         = gene_df$ENTREZID,
                OrgDb        = org.Dm.eg.db,
                keyType      = "ENTREZID",
                ont          = "CC",
                pAdjustMethod = "none",
                pvalueCutoff  = 0.05,
                readable      = TRUE)

#head(ego)

ego_df <- ego@result %>%
  arrange(p.adjust) %>%  
  slice(1:10)            

ego_df$BgRatio <- as.character(ego_df$BgRatio) 
sankeyGoPlot(goData=ego_df)
p <- sankeyGoPlot(goData=ego_df)
ggplot2::ggsave("sankey_GO.svg", plot = p, width = 8, height = 6, units = "in", dpi = 300)

ego_cc_df <- ego_cc@result %>%
  arrange(p.adjust) %>%  
  slice(1:10) 
sankeyGoPlot(goData=ego_cc_df)

#######################################################################
#Kegg Analyse

gene_ids <- unique(gene_df$ENTREZID)

ekegg <- enrichKEGG(
  gene         = gene_ids,
  organism     = "dme",
  keyType      = "ncbi-geneid",
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.2
)

ekegg <- setReadable(ekegg, OrgDb = org.Dm.eg.db, keyType = "ENTREZID")

print(head(ekegg))

#barplot(ekegg, showCategory = 20, title = "KEGG Pathway Enrichment")
#dotplot(ekegg, showCategory = 20, title = "KEGG Pathway Enrichment")
kegg_df <- ekegg@result %>%
  arrange(p.adjust) %>%  
  slice(c(1:6))            

kegg_df$BgRatio <- as.character(kegg_df$BgRatio)  
p2 <- sankeyGoPlot(goData=kegg_df)
ggplot2::ggsave("sankey_KEGG.svg", plot = p2, width = 8, height = 6, units = "in", dpi = 300)
######################################################################
library(wordcloud)
library(RColorBrewer)
library(dplyr)
library(stringr)

go_terms <- ego@result %>% arrange(p.adjust) %>% slice(1:30) %>% pull(Description)
kegg_terms <- ekegg@result %>% arrange(p.adjust) %>% slice(1:30) %>% pull(Description)

all_terms <- c(go_terms, kegg_terms)

words <- str_split(all_terms, "\\s+") |> unlist() |> str_to_lower()

stopwords <- c("of", "in", "to", "and", "the", "by", "with", "for", "from", "on", "via", "a", "an")
words_clean <- words[!words %in% stopwords]

words_clean <- words_clean[nchar(words_clean) > 2]

word_freq <- table(words_clean)

library(ggplot2)
library(ggwordcloud)

my_colors <- colorRampPalette(RColorBrewer::brewer.pal(8, "Set2"))(length(word_freq))

set.seed(123) 
word_df <- as.data.frame(word_freq)
colnames(word_df) <- c("term", "freq")
p3 <- ggplot(word_df, aes(label = term, size = freq, color = freq)) +
  geom_text_wordcloud_area(
    rm_outside = TRUE,    
    shape = "circle",     
    family = "sans",      
    eccentricity = 0.65  
  ) +
  scale_size_area(max_size = 16) + 
  scale_color_gradientn(colors = my_colors) + 
  theme_minimal(base_size = 14) +
  theme(
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"),
    legend.position = "none"
  )

ggplot2::ggsave("word_vec.svg", plot = p3, width = 8, height = 6, units = "in", dpi = 300)

