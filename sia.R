library(tidytext)
library(dplyr)
library(tidyverse)
library(wordcloud)
library(reshape2)
library(RColorBrewer)
library(readr)
library(ggplot2)
library(syuzhet)
library(stringr)
library(plotly)
library(ggpubr)
library(gridExtra)
library(tictoc)
library(tokenizers)
library(ggraph)
library(tidyr)

tic()

getwd()
setwd('C:/Users/32214609/OneDrive - Anheuser-Busch InBev/My Documents/files_sac')

file <- read_file('consolidado.txt')

file <- file %>% str_replace_all('\xf1', 'ñ')

file_df <- data.frame(file, encoding = 'latin1')

tokenized <- unnest_tokens(tbl=file_df,
                              output = "word",
                              input = "file",
                              token = "words")

tokenized %>%
  count(word, sort = TRUE) %>%
  filter(n > 50) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_text(aes(label=n), hjust= -0.2) +
  geom_col(fill = 'darkgreen', color = 'black') +
  xlab('Palabras') +
  ylab('Conteo') +
  coord_flip()

tokenized_bigrams <- unnest_tokens(tbl=file_df,
                                  output = "word",
                                  input = "file", 
                                  token = "ngrams", 
                                  n = 2)

tokenized_bigrams %>%
  count(word, sort = TRUE) %>%
  filter(n > 9) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_text(aes(label=n), hjust= -0.2) +
  geom_col(fill = 'white', color = 'black') +
  xlab('Palabras') +
  ylab('Conteo') +
  coord_flip()

tokenized_trigrams <- unnest_tokens(tbl=file_df,
                                   output = "word",
                                   input = "file", 
                                   token = "ngrams", 
                                   n = 3)

tokenized_trigrams %>%
  count(word, sort = TRUE) %>%
  filter(n > 4) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_text(aes(label=n), hjust= -0.2) +
  geom_col(fill = 'red', color = 'black') +
  xlab('Palabras') +
  ylab('Conteo') +
  coord_flip()

tokenized_2 <- get_tokens(file)

#debug <- tokenize_ngrams(file, n=2)

#debug_2 <- unlist(debug)

sia <- get_nrc_sentiment(debug_2, lang='spanish')

summary(sia)

palabras_anger<- data.frame(tokenized_2[sia$anger > 0])
palabras_disgust <- data.frame(tokenized_2[sia$disgust > 0])
palabras_fear <- data.frame(tokenized_2[sia$fear > 0])
palabras_joy <- data.frame(tokenized_2[sia$joy > 0])
palabras_sadness <- data.frame(tokenized_2[sia$sadness > 0])
palabras_surprise <- data.frame(tokenized_2[sia$sadness > 0])
palabras_trust <- data.frame(tokenized_2[sia$trust > 0])
palabras_negative <- data.frame(tokenized_2[sia$negative > 0])
palabras_positive <- data.frame(tokenized_2[sia$positive > 0])

group_positive <- palabras_positive %>% group_by(tokenized_2.sia.positive...0.) %>% summarise(n = n())

plot_positive <- group_positive %>%
    filter(n > 10) %>%
    mutate(tokenized_2.sia.positive...0. = reorder(tokenized_2.sia.positive...0., n)) %>%
    ggplot(aes(tokenized_2.sia.positive...0., n)) +
    geom_text(aes(label=n), hjust= -0.2) +
    geom_col(fill = 'darkgreen') +
    xlab('Palabras') +
    ylab('Conteo') +
    ggtitle('Positivo') +
    theme(
      plot.title = element_text(hjust=0.5)
    ) +
    coord_flip()

group_joy <- palabras_joy %>% group_by(tokenized_2.sia.joy...0.) %>% summarise(n = n())

plot_joy <- group_joy %>%
  filter(n > 5) %>%
  mutate(tokenized_2.sia.joy...0. = reorder(tokenized_2.sia.joy...0., n)) %>%
  ggplot(aes(tokenized_2.sia.joy...0., n)) +
  geom_text(aes(label=n), hjust= -0.2) +
  geom_col(fill = 'yellow') +
  xlab('Palabras') +
  ylab('Conteo') +
  ggtitle('Alegría') +
  theme(
    plot.title = element_text(hjust=0.5)
  ) +
  coord_flip()

group_fear <- palabras_fear %>% group_by(tokenized_2.sia.fear...0.) %>% summarise(n = n())

plot_fear <- group_fear %>%
  filter(n > 3) %>%
  mutate(tokenized_2.sia.fear...0. = reorder(tokenized_2.sia.fear...0., n)) %>%
  ggplot(aes(tokenized_2.sia.fear...0., n)) +
  geom_text(aes(label=n), hjust= -0.2) +
  geom_col(fill = 'red') +
  ggtitle('Miedo') +
  theme(
    plot.title = element_text(hjust=0.5)
  ) +
  xlab('Palabras') +
  ylab('Conteo') +
  coord_flip()

group_trust <- palabras_trust %>% group_by(tokenized_2.sia.trust...0.) %>% summarise(n = n())

plot_trust <- group_trust %>%
  filter(n > 10) %>%
  mutate(tokenized_2.sia.trust...0. = reorder(tokenized_2.sia.trust...0., n)) %>%
  ggplot(aes(tokenized_2.sia.trust...0., n)) +
  geom_text(aes(label=n), hjust= -0.2) +
  geom_col(fill = 'green') +
  xlab('Palabras') +
  ylab('Conteo') +
  ggtitle('Confianza') +
  theme(
    plot.title = element_text(hjust=0.5)
  ) +
  coord_flip()

group_sadness <- palabras_sadness %>% group_by(tokenized_2.sia.sadness...0.) %>% summarise(n = n())

plot_sadness <- group_sadness %>%
  filter(n > 5) %>%
  mutate(tokenized_2.sia.sadness...0. = reorder(tokenized_2.sia.sadness...0., n)) %>%
  ggplot(aes(tokenized_2.sia.sadness...0., n)) +
  geom_text(aes(label=n), hjust= -0.2) +
  geom_col(fill = 'blue') +
  xlab('Palabras') +
  ylab('Conteo') +
  ggtitle('Tristeza') +
  theme(
    plot.title = element_text(hjust=0.5)
  ) +
  coord_flip()

group_negative <- palabras_negative %>% group_by(tokenized_2.sia.negative...0.) %>% summarise(n = n())

plot_negative <- group_negative %>%
  filter(n > 5) %>%
  mutate(tokenized_2.sia.negative...0. = reorder(tokenized_2.sia.negative...0., n)) %>%
  ggplot(aes(tokenized_2.sia.negative...0., n)) +
  geom_text(aes(label=n), hjust= -0.2) +
  geom_col(fill = 'red') +
  xlab('Palabras') +
  ylab('Conteo') +
  ggtitle('Negativo') +
  theme(
    plot.title = element_text(hjust=0.5)
  ) +
  coord_flip()

group_disgust <- palabras_disgust %>% group_by(tokenized_2.sia.disgust...0.) %>% summarise(n = n())

plot_disgust <- group_disgust %>%
  filter(n > 2) %>%
  mutate(tokenized_2.sia.disgust...0. = reorder(tokenized_2.sia.disgust...0., n)) %>%
  ggplot(aes(tokenized_2.sia.disgust...0., n)) +
  geom_text(aes(label=n), hjust= -0.2) +
  geom_col(fill = 'purple') +
  xlab('Palabras') +
  ylab('Conteo') +
  ggtitle('Asco') +
  theme(
    plot.title = element_text(hjust=0.5)
  ) +
  coord_flip()

group_surprise <- palabras_surprise %>% group_by(tokenized_2.sia.sadness...0.) %>% summarise(n = n())

plot_surprise <- group_surprise %>%
  filter(n > 5) %>%
  mutate(tokenized_2.sia.sadness...0. = reorder(tokenized_2.sia.sadness...0., n)) %>%
  ggplot(aes(tokenized_2.sia.sadness...0., n)) +
  geom_text(aes(label=n), hjust= -0.2) +
  geom_col(fill = 'black') +
  xlab('Palabras') +
  ylab('Conteo') +
  ggtitle('Sorpresa') +
  theme(
    plot.title = element_text(hjust=0.5)
  ) +
  coord_flip()

grid.arrange(plot_positive, plot_negative, ncol = 2)

grid.arrange(plot_joy, plot_sadness, plot_trust, plot_fear, plot_surprise, plot_disgust, nrow = 2, ncol = 3)
toc()

# Gráfica de correlación

# Bigramas

set.seed(1000)

tokenized_count_bigram <- tokenized_bigrams %>%
  count(word, sort = TRUE) %>%
  filter(n > 9) %>%
  mutate(word = reorder(word, n))

bigrams_separated <- tokenized_count_bigram %>%
  separate(word, c("word1", "word2"), sep = " ")

a <- grid::arrow(type = "closed", length = unit(.15, "inches"))

ggraph(bigrams_separated, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE,
                 arrow = a, end_cap = circle(.07, 'inches')) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void()

# Trigramas

set.seed(2000)

tokenized_count_trigram <- tokenized_trigrams %>%
  count(word, sort = TRUE) %>%
  filter(n > 9) %>%
  mutate(word = reorder(word, n))

trigrams_separated <- tokenized_count_trigram %>%
  separate(word, c("word1", "word2"), sep = " ")

ggraph(trigrams_separated, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE,
                 arrow = a, end_cap = circle(.07, 'inches')) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void()

 # ggarrange(plot_joy, plot_sadness, plot_trust, plot_fear, plot_surprise, plot_disgust, rremove("x.text"),
  #        nrow = 3, ncol = 2)
