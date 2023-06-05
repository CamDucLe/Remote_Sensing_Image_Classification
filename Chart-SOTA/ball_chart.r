# Libraries
library(ggplot2)
library(dplyr)
library(magrittr)

# The dataset is provided in the gapminder library
#library(ballchart)
#data <- ballchart 
#geom_text(aes(label=model)) +

library(readxl)
sota <- read_excel("C:/Users/HP/Downloads/sota.xlsx")
data <- sota

# Most basic bubble plot
data %>%
  ggplot(aes(x=para, y=acc, size=para, color=model))+
  scale_y_continuous(limits=c(87,97))+
  scale_x_continuous(limits=c(1,150))+
  #theme(axis.title.x=element_text(size=40,face="bold"))+
  geom_point(alpha=0.9, fill = "white")+ 
  theme_linedraw(base_size = 19)+ 
  #scale_size(range = c(0.1, 15), name="Parameter (M)")
  geom_text(aes(label=model), 
            size = c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,5,5),
            fontface='bold',
            hjust = 0, 
            nudge_x = c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-6,-5,-2.7), 
            nudge_y = c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,-0.75,-0.35)) +
  xlab("Parameter (M) ")+
  ylab("Accuracy (%) ")+
  
 
  scale_size_area(max_size = 15,name="Parameter (M)")





  
  