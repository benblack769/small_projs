require(ggplot2)

in_file_name = "operate_output.txt"

data = read.table(file=in_file_name,sep='\t',header=FALSE)

data$Type = ifelse(data$V1 == 0,"random","weighted")

data$Length = data$V7

ggplot(data,aes(x=factor(Type),y=Length))+
  geom_boxplot() + 
  scale_y_log10() 
