
library(devtools)

# install python and tensorflow/keras
install.packages("reticulate")
install.packages("keras")
library(reticulate)
reticulate::install_python()
library(keras)
install_keras()

# Check and install the package
getwd()
path_to_packages <- "path_to_package_folder"
devtools::check(path_to_packages)
devtools::install(pkg = path_to_packages, reload = TRUE)
