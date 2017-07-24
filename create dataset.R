# install.packages('magick')
library(magick) 
library(digest)
library(data.table)
library(parallel)
#str(magick::magick_config())

# try to create a set I can play with.
chr.target.dir <- 'photos'

dir.create(chr.target.dir, showWarnings = FALSE, recursive = TRUE)


lst.folders <- list(
  bad = c('Bad photos directories here',
          'and here'),
  good = c('Good photos directories here'),
  other_label = c('Some other label, if you wish')
  )

lst.folders <- lst.folders[c('good','bad')] # exclude any other labels

lst.files <- lapply(lst.folders, function(y) {
  chr.files <- lapply(y, function(x) {
    if(dir.exists(x)) list.files(x, recursive = TRUE, full.names = TRUE, pattern = '\\.jp[e]?g$') else character(0)
  })
  chr.files <- unlist(chr.files)
  return(chr.files)
})

dt.labels <- fread('photos/labels.csv')
lst.files <- lapply(lst.files, setdiff, y = dt.labels$original)

# process them all, dump into two folders
# then remove potential dupes, I guess
resizePhoto <- function(chr.file.original, chr.label, chr.target.directory = chr.target.dir, int.width = 480) {
  img.full.size <- image_read(chr.file.original)
  chr.hash <- digest(chr.file.original)
  #img.full.size
  img.resized <- image_scale(img.full.size, as.character(int.width))
  chr.file.target <- paste0(chr.target.directory, '/', chr.hash, '.jpg')
  image_write(img.resized, chr.file.target, format = 'jpeg', quality = 100)
  
  lst.out <- list(
    original = chr.file.original, 
    resized = chr.file.target, 
    width = int.width,
    height = image_info(img.resized)$height,
    hash = chr.hash,
    label = chr.label
  )
  rm(img.full.size)
  rm(img.resized)
  return(lst.out)
}

lst.add <- lapply(seq_along(lst.files), function(i) {
  cat('\n', as.character(Sys.time()), ': ', 
    'working on ', names(lst.files)[i],
    '\n', 
    sep = '')
  chr.files <- lst.files[[i]]

  pb <- txtProgressBar(min = 0, max = length(chr.files), style = 3)
  
  dt.add <- rbindlist(lapply(seq_along(chr.files), function(j) {
    if(j == 1) cat('\n', as.character(Sys.time()), ': ', 
                            "Starting",
                            '\n', 
                            sep = '')
    
    setTxtProgressBar(pb, j)
    return(resizePhoto(chr.files[j], chr.label = names(lst.files)[i]))
  }))
  # too memory intensive for mclapply.
  close(pb)
  return(dt.add)
})

dt.add <- rbindlist(lst.add)
dt.all <- rbind(dt.labels,
                dt.add)

fwrite(dt.all, paste0(chr.target.dir, '/labels_new.csv'))

