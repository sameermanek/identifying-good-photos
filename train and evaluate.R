library(keras)
library(ggplot2)
library(scales)
library(data.table)
library(jpeg)
library(stringr)
library(pROC)
library(plyr)

use_condaenv('r-tensorflow', conda = '/anaconda/bin/conda')

isRstudio <- function(){
  if(!exists("RStudio.Version"))return(FALSE)
  if(!is.function(RStudio.Version))return(FALSE)
  return(TRUE)
}

# Improve this logging
# Should probably use `message` rather than cat.
logNote <- function(...) {
  chr.note <- paste0(as.character(Sys.time()), ': ', paste(...))
  if(!parallel:::isChild()) cat(chr.note, '\n', sep = '')
  
  # This could still fail if in rstudio in an inaccessible directory.
  if(parallel:::isChild() && isRstudio()) {
    # create a log file, write to that
    if(!file.exists('log')) {
      file.create('log')
    }
    write.table(data.frame(log = chr.note),
                file = 'log',
                row.names = FALSE,
                col.names = FALSE,
                append = TRUE,
                quote = FALSE)
  }
}



# Parameters
dt.images <- fread('photos/labels_new.csv')
int.log.interval <- 1L
int.batch.size <- 4L
int.batches <- 5000L

dt.images[, filename:=str_match(original, '^.*/(.+?$)')[, 2]]
dt.images[, .N, filename][N > 1]
# Dedupe (based on same image name across multiple directories)
setorder(dt.images, filename, -label) # preserve the 'good' labeled dupes
dt.images <- unique(dt.images, by = 'filename')

# Take a character vector of image files (jpegs), return a list (same order) of image matrices (3d each)
getImages <- function(chr.filenames, int.width = NA_integer_, int.height = NA_integer_, int.channels = 3) {
  if(is.na(int.width) || is.na(int.height)) {
    lst.images <- lapply(chr.filenames, readJPEG)
    return(lst.images)
  } else {
    arr.images <- vapply(chr.filenames, function(x) {
      arr.image <- tryCatch(
        readJPEG(x), 
        error = function(e) {
          warning(e)
          logNote('Issue with image', chr.filenames)
          array(dim = c(int.height, int.width, int.channels), data = NA_real_)
        })
      return(arr.image)
    }, array(dim = c(int.height, int.width, int.channels), data = NA_real_))
    arr.images <- aperm(arr.images, perm = c(4,1,2,3)) # I think this is what I want
    return(arr.images)
  }
}

# arbitrarily take 80% as training, 10% as validation, 10% as test set
dt.images[, set:=sample(c('training','validation', 'test'), replace = TRUE, size = .N, prob = c(0.80, 0.10, 0.10))]
dt.images[, label:=factor(label, levels = sort(unique(label)))]

# Remove any monochrome (1-channel) images
lst.validation <- lapply(dt.images$resized, function(x) {a <- readJPEG(x); list(dimensions = length(dim(a)), pts = length(a))})
dt.validation <- rbindlist(lst.validation)
which(dt.validation$dimensions != 3)
dt.images <- dt.images[dt.validation$dimensions == 3]

# Get a constant-dimension batch (as a 4d array).
# Return a list with: 1) 4d array (x's) and 2) matrix of labels
getBatch <- function(int.batch.size = 4L,
                     chr.set = 'training',
                     log.rebalance = TRUE, 
                     dt.table = dt.images) {
  
  # First, sample a height
  int.height <- dt.table[set == chr.set, sample(height,1)]
  
  # Subset to appropriate rows
  dt.table <- dt.table[height == int.height & set == chr.set]
  
  # apply probabilities as appropriate
  dt.table[, probability:=ifelse(log.rebalance, nrow(dt.table)/.N,1), by = list(label)]
  
  # Sample with replacement
  dt.table <- dt.table[sample(1:.N, size = int.batch.size, replace = TRUE, prob = probability)]
  
  arr.images <- getImages(dt.table$resized, int.width = dt.table$width[1], int.height = dt.table$height[1], int.channels = 3)
  
  mat.y <- to_categorical(
    matrix(as.integer(dt.table$label) - 1L, nrow = nrow(dt.table)),
    num_classes = length(levels(dt.table$label))
  )
  return(list(
    x = arr.images,
    y = mat.y
  ))
  
}

# Define the model. Allow variable height and batch sizes
model <- keras_model_sequential()

model %>%
  layer_conv_2d(
    filters = 32,
    kernel_size = c(3,3),
    padding = 'same',
    input_shape = list(NULL, 480, 3) # this is the risky part.
  ) %>%
  layer_activation("relu") %>%
  
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", strides = c(2L, 2L)) %>%
  layer_activation("relu") %>%
  
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", strides = c(2L,2L)) %>%
  layer_activation("relu") %>%

  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", strides = c(2L,2L)) %>%
  layer_activation("relu") %>%

  layer_conv_2d(filter = 64, kernel_size = c(3,3), padding = "same", strides = c(2L,2L)) %>%
  layer_activation("relu") %>% # Massively increase the logical field 
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_global_average_pooling_2d() %>% 
  layer_activation('relu') %>%
  
  layer_dense(length(levels(dt.images$label))) %>%
  layer_activation('softmax')
# Increased dimensionality a little, increased max field size a bunch, increased complexity a few times


opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)

model %>% compile(
  loss = "binary_crossentropy", 
  optimizer = opt,
  metrics = "accuracy"
)

# evaluate validation/test sets
# for now, hard-coded metrics. Could improve that logic a bit
evaluatePerformance <- function(m, 
                                chr.set = 'validation', 
                                dt.table = dt.images,
                                num.threshold = NA_real_) {
  dt.table <- dt.table[set == chr.set]
  logNote('Making predictions')
  # Should I rebalance the validation set? It seems like the metrics could be 'deceptive' here
  mat.prediction <- vapply(1:nrow(dt.table), function(i) {
    arr.image <- dt.table[i, getImages(resized, int.width = width, int.height = height, int.channels = 3)]
    
    arr.res <- m %>%
      predict_on_batch(x = arr.image)
    
    return(arr.res[1, ])
  }, numeric(2)) 
  mat.prediction <- t(mat.prediction)
  
  mat.y.test <- to_categorical(
    matrix(as.integer(dt.table$label) - 1L, nrow = nrow(dt.table)),
    num_classes = length(levels(dt.table$label))
  )
  
  obj.roc <- roc(mat.y.test[, 2], 
                 mat.prediction[, 2])
  
  dt.roc <- data.table(
    true_positive = obj.roc$sensitivities,
    true_negative = obj.roc$specificities,
    false_positive = 1-obj.roc$specificities,
    false_negative = 1-obj.roc$sensitivities,
    threshold = obj.roc$thresholds)
  
  dt.roc[, f1:=(2*true_positive) / (2*true_positive + false_negative + false_positive)]
  if(is.na(num.threshold)) num.threshold <- dt.roc[which.max(f1), threshold]
  
  # Maybe introduce the loss here too?
  return(list(roc = dt.roc,
              auc = obj.roc$auc,
              acc = mean((mat.prediction[, 2] >= num.threshold) == mat.y.test[, 2]),
              positive_predictions = mean((mat.prediction[, 2] >= num.threshold)),
              f1 = max(dt.roc$f1),
              prediction = mat.prediction,
              threshold = num.threshold
  ))
}

lst.res <- list()
for(i in 1:int.batches) {
  # Wish I could use tensorboard logging here. @todo
  if(i %% int.log.interval == 0) cat(as.character(Sys.time()), i, '\n')
  lst.sample <- getBatch(int.batch.size = int.batch.size, chr.set = 'training', log.rebalance = TRUE, dt.table = dt.images)
  
  lst.metric <- model %>%
    train_on_batch(x = lst.sample[['x']],
                   y = lst.sample[['y']])
  
  names(lst.metric) <- model$metrics_names
  
  if(i %% 100 == 0 | i == 1) {
    lst.validation <- evaluatePerformance(m = model, chr.set = 'validation', dt.table = dt.images)
    lst.metric[['validation']] <- lst.validation
  }
  
  lst.res[[i]] <- lst.metric
  
  if(i %% 100 == 0 | i == 1) {
    dt.res <- rbindlist(lapply(lst.res, function(x) {
      list(
        loss = x[['loss']],
        acc = x[['acc']])
    }))
    dt.res[, batch:=1:.N]
    dt.res[, set:='train']
    
    dt.validation <- rbindlist(lapply(seq_along(lst.res), function(j) {
      if('validation' %in% names(lst.res[[j]])) {
        return(list(
          acc = lst.res[[j]][['validation']][['acc']],
          auc = lst.res[[j]][['validation']][['auc']],
          f1 = lst.res[[j]][['validation']][['f1']],
          batch = j,
          set = 'validation'))
      } else {
        return(NULL)
      }
    }))
    
    dt.plot <- rbind(
      melt.data.table(dt.res, id.vars = c('set','batch')),
      melt.data.table(dt.validation, id.vars = c('set','batch'))
    )
    
    setkey(dt.plot, batch, set, variable)
    setorder(dt.plot, batch, set, variable)
    library(TTR)
    dt.plot[set == 'train', ma:=if(i == 1) value else SMA(value, ceiling(i/4)), by = list(variable)]
    a <- ggplot(dt.plot, aes(batch,value, colour = set)) + 
      facet_grid(variable ~ ., scale = 'free_y') + 
      theme_bw() 
    if(i <= 1e3) a <- a + geom_line(alpha = 0.2)
    if(i == 1) a <- a + geom_point()
    a <- a +
      geom_line(aes(y = ma)) + 
      geom_line(data = dt.plot[set == 'validation'])
    #suppressWarnings({print(a)})
    suppressWarnings({ggsave(a, file = paste0('graphs/performance_', i, '.pdf'), width = 12, height = 8)})
  }
  # save model checkpoints?
  if(i %% 100 == 0) {
    dir.create('checkpoints/', showWarnings = FALSE)
    model %>%
      save_model_hdf5(filepath = paste0('checkpoints/train_v3.1_', i, '.h5'))
    fwrite(dt.plot, 'checkpoints/log.csv')
  }
  
}

##################
# Evaluate results

chr.checkpoints <- list.files('checkpoints/', pattern = 'train_v3\\.1_.*000.*\\.h5$', full.names = T) # limited to the thousands
chr.checkpoints <- chr.checkpoints[order(as.numeric(str_extract(chr.checkpoints, '[0-9]+0+')))]

lst.checkpoint.performance <- lapply(chr.checkpoints, function(x) {
  logNote(x)
  m <- load_model_hdf5(x)
  
  logNote('Evaluating Validation Set')
  lst.validation <- evaluatePerformance(
    m, 
    'validation', 
    dt.images, 
    NA_real_ # let it choose/overfit on the threshold
  )
  
  logNote('Evaluating Test Set')
  lst.test <- evaluatePerformance(
    m, 
    'test', 
    dt.images, 
    lst.validation[['threshold']]
  )

  logNote('Evaluating Training Set')
  lst.train <- evaluatePerformance(
    m,
    'training',
    dt.images,
    NA_real_ # let it choose/overfit on the threshold
  )
  

  
  return(list(checkpoint = x,
              training = lst.train,
              validation = lst.validation,
              test = lst.test
  ))
})

dt.checkpoint <- rbindlist(lapply(lst.checkpoint.performance, function(x) {
  rbindlist(lapply(setdiff(names(x), 'checkpoint'), function(y) {
    data.table(set = y, 
               checkpoint = x[['checkpoint']],
               acc = x[[y]][['acc']],
               auc = x[[y]][['auc']],
               f1 = x[[y]][['f1']])
  }))
}))
dt.checkpoint[, checkpoint_number:=as.numeric(str_extract(checkpoint, '[0-9]+0+'))]

dt.plot <- melt.data.table(dt.checkpoint, id.vars = c('set','checkpoint_number'), measure.vars = c('acc','auc','f1'))

ggplot(dt.plot, aes(checkpoint_number, value, colour = set)) + 
  geom_point() + 
  geom_line() + 
  facet_grid(variable ~ ., scale = 'free_y') + 
  theme_bw()


library(scales)
ggplot(lst.test$roc, aes(false_positive, true_positive)) + 
  geom_abline(slope = 1, intercept = 0, linetype = 2, colour = 'grey') + 
  geom_step(size = 2) + 
  theme_bw() + 
  coord_equal() + 
  labs(x = 'False Positive Rate', y = 'True Positive Rate', title = 'Test Set', caption = paste0('ROC Curve for Test set, ~200 images\n', 'AUC = ', round(lst.test$auc,2)))




# Again... I really should be transfer learning given the small-ish sample sizes I have.
