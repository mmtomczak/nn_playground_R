#' Class R6 representing a TensorFlow model
#'
#' @author Mateusz Tomczak
#' @docType class
#' @importFrom R6 R6Class
#' @import keras
#' @import purrr
#' @export tfmodel
#' @return Object of \code{R6Class} with methods allowing to build a simple neural network
#' @format \code{R6Class} object.
#' @examples
#' tfmodel$new("sample_model", 7, 3, data.frame(x=list(1,2), y=list(10,20)), 0.2)
#' tfmodel$new("my_model", 4, 3, iris, 0.15)
#' @field name Model name.
#' @field layers List of layers that will be used to build a neural network.
#' @field input_shape Input dimension equal to number of features in the input.
#' @field output_shape Output dimension equal to the number of unique target values.
#' @field model Compiled TensorFlow model.
#' @field history Training history data by epoch.
#' @field data Dataset that will be used by the model
#' @field test_split Percentage of the observations that will be used as a test set/validation set
#'
#' @section Important Information:
#' \describe{
#'   \item{\code{Package dependencies}}{To run package prior installation of python and keras for R is required. To do this, use the following commands: \cr install.packages("reticulate") \cr install.packages("keras") \cr reticulate::install_python() \cr library(keras) \cr install_keras()}
#' }
#'
#' @section Methods:
#' \describe{
#'   \item{\code{new(name, input_shape, output_shape, data, test_split)}}{Method used to create \code{tfmodel} class object. Values of \code{input_shape} and \code{output_shape} are integers used to create layers stored in fields \code{inputs} and \code{outputs}}
#'   \item{\code{add_layer(layer_type, units, act_func)}}{Method used to add a new layer to the model. Argument \code{layer_type} may be "Dense" or "Dropout" with \code{units} number of neurons for "Dense" layer and \code{act_func} activation function ("relu", "linear" or "elu") or \code{units} fraction of randomly selected neurons dropped for "Dropout" layer (in this case \code{act_func} argument will be ignored)}
#'   \item{\code{build_model(optimizer, lr)}}{Builds and compiles the model with \code{optimizer} optimizer function ("adam", "adamax", "sgd" or "ftrl") with \code{lr} learning rate}
#'   \item{\code{train(epochs)}}{Trains the compiled model for \code{epochs} number of epochs. Requires using \code{build_model(optimizer, lr)} before.}
#'   \item{\code{predict(x)}}{Predicts the labels for x. Requires trained model.}
#'   \item{\code{train_test_split_data}}{Splits the field \code{data} into train and test datasets. Used by \code{predict(x)} method. Returns list of X_train, X_test, y_train, y_test created from \code{data}}
#'   \item{\code{add_layers_from_df(layers_df)}}{Adds layers stored in rows of \code{layers_df} data frame into the model.}
#' }

tfmodel <- R6Class("tfmodel",
                   public = list(
                     name = "model_name",
                     layers = list(),
                     input_shape = NULL,
                     model = NULL,
                     history = NULL,
                     output_shape = NULL,
                     data = NULL,
                     test_split = NULL,

                     initialize = function(name, input_shape, output_shape, data, test_split){
                       if (!is.character(name)) {
                         warning("Model name is not a string!")
                       }
                       if (!is.numeric(input_shape)) {
                         stop("Input shape must be an integer!")
                       }
                       if (!is.numeric(output_shape)) {
                         stop("Output shape must be an integer!")
                       }
                       if (!is.numeric(test_split) | test_split >= 1 | test_split <= 0) {
                         stop("Train/Test Split must be a number between 0 and 1!")
                       }
                       self$name = name
                       self$input_shape = input_shape
                       self$output_shape = output_shape
                       self$data = data
                       self$test_split = test_split

                       invisible(self)
                     },

                     add_layer = function(layer_type, units, act_func){
                       if (act_func=="") {
                         if (!(layer_type == "Dropout")){
                           stop("Only dropout layer does not need activation function specified!")
                         }
                         if (!is.numeric(units) || units <= 0 | units >= 1){
                           stop("Rate for dropout must be between 0 and 1!")
                         }
                         layer = keras::layer_dropout(rate=units)
                       }
                       else{
                         if (!(layer_type=="Dense") & !(layer_type=="Dropout")) {
                           stop("Layer type not recognized!")
                         }
                         if (!is.numeric(units)){
                           stop("Layer units must be an integer!")
                         }
                         if (!(act_func %in% c("relu", "linear", "elu"))) {
                           stop("Activation function not recognized!")
                         }

                         if (layer_type == "Dense") {
                           layer = keras::layer_dense(units=units, activation=act_func)
                         }
                       }

                       self$layers = append(self$layers, layer)
                     },

                     build_model = function(optimizer, lr) {
                       if (missing(lr)) {
                         lr = 0.001
                       }

                       if (!is.numeric(lr) | lr <= 0 | lr >= 1) {
                         stop("Learning rate must be a number between 0 and 1")
                       }
                       if (optimizer == "Adam") {
                         optimizer_func = keras::optimizer_adam(learning_rate = lr)
                       }
                       else if (optimizer == "SGD") {
                         optimizer_func = keras::optimizer_sgd(learning_rate = lr)
                       }
                       else if (optimizer == "Adamax") {
                         optimizer_func = keras::optimizer_adamax(learning_rate = lr)
                       }
                       else if (optimizer == "FTRL") {
                         optimizer_func = keras::optimizer_ftrl(learning_rate = lr)
                       }
                       else {
                         warning("Could not recognize optimizer! Defaulting to Adam")
                         optimizer_func = keras::optimizer_adam(learning_rate = lr)
                       }

                       input_layer = keras::layer_input(shape=self$input_shape, name="input_layer")
                       output_layer = keras::layer_dense(units=self$output_shape, activation="sigmoid", name = "output_layer")

                       outputs = input_layer

                       for (layer in self$layers){
                         outputs = outputs %>% layer
                       }

                       outputs = outputs %>% output_layer

                       self$model = keras::keras_model(input_layer, outputs)

                       self$model %>% keras::compile(
                         loss="sparse_categorical_crossentropy",
                         optimizer = optimizer_func
                       )
                     },

                     train = function(epochs) {
                       data <- self$train_test_split_data()

                       if (is.null(self$model)) {
                         stop("Model not compiled - compile model first using build_model()")
                       }
                       if (!is.numeric(epochs)) {
                         stop("Number of epochs must be a number!")
                       }
                       else {
                         self$history = self$model %>% keras::fit(as.matrix(data[[1]]),
                                                           as.matrix(data[[3]]),
                                                           epochs=epochs,
                                                           validation_data=list(as.matrix(data[[2]]), as.matrix(data[[4]])),
                                                           verbose=0)
                       }
                     },

                     predict = function(x) {
                       if (is.null(self$model)) {
                         stop("Model has not yet been compiled - compile the model first using build_model()!")
                       }
                       if (is.null(self$history)) {
                         stop("Model has not yet been trained - train model first!")
                       }
                       preds <- predict(self$model, x, verbose=0)
                       return(preds)
                     },

                     train_test_split_data = function() {
                       df = self$data
                       frac = self$test_split
                       if (!is.data.frame(df)) {
                         stop("Data must be in data frame format!")
                       }
                       if (!is.numeric(frac) | frac <= 0 | frac >= 1) {
                         stop("Frac must be a number between 0 and 1!")
                       }
                       bound = floor(nrow(df) * frac)
                       df = as.data.frame(df[sample(nrow(df)), ])
                       y = df[ncol(df)]
                       X = df[1:(ncol(df)-1)]
                       y_test = y[1:bound,]
                       y_train = y[(bound + 1):nrow(y),]

                       X_test = X[1:bound,]
                       X_train = X[(bound + 1):nrow(X),]
                       return(list(X_train, X_test, y_train, y_test))
                     },

                     add_layers_from_df = function(layers_df) {
                       purrr::pmap(layers_df, self$add_layer)
                     }
                   ))


