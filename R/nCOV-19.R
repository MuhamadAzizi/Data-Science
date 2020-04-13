# Import library

library(dplyr)
library(ggplot2)

# Import dataset

dataset <- read.csv("../dataset/2019_nC0v_20200121_20200126 - SUMMARY_FIX_MISS.csv")

# Create dataset based on day of date in dataset

dataset$Date.last.updated <- as.Date(dataset$Date.last.updated) # Convert Date.last.updated into date format

group_day <- dataset %>% group_by(date = as.Date(cut(Date.last.updated, breaks = "day"))) %>%
  summarise(total_confirmed = sum(Confirmed),
            total_suspected = sum(Suspected),
            total_recovered = sum(Recovered),
            total_death = sum(Deaths))

# Split group_day into total_confirmed, total_suspected, total_recovered, total_death

total_confirmed <- group_day %>% select(value = total_confirmed) %>% mutate(type = "total_confirmed")
total_suspected <- group_day %>% select(value = total_suspected) %>% mutate(type = "total_suspected")
total_recovered <- group_day %>% select(value = total_recovered) %>% mutate(type = "total_recovered")
total_death <- group_day %>% select(value = total_death) %>% mutate(type = "total_death")

# Bind rows of total_confirmed, total_suspected, total_recovered, total_death inro all_total

all_total <- bind_rows(total_confirmed, total_suspected, total_recovered, total_death)

# Create multiple value of date to synchronize date date and all_total

date <- bind_rows(select(group_day, date), select(group_day, date), select(group_day, date), select(group_day, date))

# Bind cols of date and all_total

data <- bind_cols(date, all_total)

# Create bar chart

plot <-
  ggplot(data, aes(x = date, y = value)) + geom_bar(stat = 'identity', aes(fill = type)) +
  geom_text(aes(label = value, vjust = -0.5), size = 3) +
  facet_wrap(~type) +
  labs(
    title = "Comparison of nCov-19",
    subtitle = "Comparison of nCov-19 at Jan 22 - Jan 26, 2020",
    fill = ""
  ) +
  xlab("Date") + ylab("Value") +
  ylim(0, 6000) + # Range of y-axis
  theme(
    plot.title = element_text(color = '#424242', size = 14),
    plot.subtitle = element_text(color = '#999999', size = 10),
    legend.position = 'top'
  )

plot