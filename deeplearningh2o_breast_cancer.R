library(mlbench)
library(h2o)


h2o.init()


data(BreastCancer)

data<-BreastCancer[,-1]

data[,c(1:ncol(data))]<-sapply(data[,c(1:ncol(data))],as.numeric)

data[,'Class']<-as.factor(data[,'Class'])

splitSample<-sample(1:3,size = nrow(data),prob = c(0.6,0.2,0.2),replace=T)


train_h2o<-as.h2o(data[splitSample==1,])
val<-as.h2o(data[splitSample==2,])
test<-as.h2o(data[splitSample==3,])


model<-h2o.deeplearning(x=1:9,
                        y=10,
                        training_frame = train_h2o,
                        activation = 'TanhWithDropout',
                        input_dropout_ratio = 0.2,
                        balance_classes = T,
                        hidden = c(10,10),
                        hidden_dropout_ratios = c(0.3,0.3),
                        epochs = 10,
                        seed = 0)
?h2o.deeplearning
h2o.confusionMatrix(model)

h2o.confusionMatrix(model,val)

hyper_params<-list(
  activation =c("Tanh", "TanhWithDropout", "Rectifier",
                "RectifierWithDropout", "Maxout", "MaxoutWithDropout"),
  hidden = list(c(20,20),c(30,30,30),c(50,50),c(25,25,25,25)),
  input_dropout_ratio=c(0,0.05),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)

search_criteria=list(
  strategy = 'RandomDiscrete',max_runtime_secs=600,
  max_models=100,seed=123456,stopping_rounds=5,
  stopping_tolerance=0.00001,stopping_metric= "AUTO" 
)

dl_random_grid<-h2o.grid(
  algorithm = "deeplearning",
  grid_id = "dl_grid_random",
  training_frame = train_h2o,
  validation_frame=val,
  x=1:9,
  y=10,
  epochs=1,
  stopping_metric="logloss",
  stopping_tolerance=1e-2,
  stopping_rounds=2,
  hyper_params = hyper_params,
  search_criteria = search_criteria
)


grid<-h2o.getGrid("dl_grid_random",sort_by = "logloss",decreasing = F)

grid@summary_table[1,]

best_model<-h2o.getModel(grid@model_ids[[1]])

best_model
h2o.confusionMatrix(model,test)

h2o.shutdown()
