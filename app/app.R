#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

library(tfmodel)
library(shiny)
library(shinyjs)
library(bslib)
library(tidyverse)
library(keras)

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Neural Network Training App"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
          style="height:100%",
          width = "3",
          fileInput(
            inputId = "file",
            label = "Choose data to train the model",
            multiple = FALSE,
            accept = c(".csv"),
            width = "300px",
            buttonLabel = icon("folder"),
            placeholder = "Select file..."
          )
        ),
        
        mainPanel(
          tabsetPanel(
            tabPanel("Introdunction",
                     br(),
                     page_fluid(
                       layout_columns(
                         card(
                           h2("Interacive Neural Network App"),
                           p("Goal of this application is to introduce a basic understanding on how to create Neural Network models and how they work."),
                           p("Model creation is based on TensorFlow library, native to Python language. R implementation is based on the Python version, thus learning TensorFlow in Python can allow you to use it in R, and vice versa (after getting used to the slightly different syntax)"),
                           h6("As this is an application that focuses mainly on the Neural Network building, some assumptions about uploaded data must be met for application to work correctly, namely:"),
                           HTML("<ul>
                                  <li>Only classification data is allowed (with 2 or more targets)</li>
                                  <li>Only .csv file format is allowed</li>
                                  <li>All columns must be in numeric format</li>
                                  <li>Target column is the last column in the csv file</li>
                                  <li>Target values are integers, starting from 0, up to n, were n is equal to the number of possible targets minus 1</li>
                                  <li>There is no limit on the number of explanatory variables</li>
                                  <li>No missing values are present in the dataset</li>
                                  <li>If needed, uploaded data is normalized. No normalization is applied by the app</li>
                                </ul>")
                           )
                         )
                       )
                     ),
            tabPanel("Build the model", 
                      br(),
                     
                      tabsetPanel(
                        tabPanel(
                          "Add layers",
                          br(),
                          span(textOutput('layer_no_data'), style="color:red;padding:10px;display:block;text-align:center", width="100%"),
                          page_fluid(
                            layout_columns(
                              card(
                                div(style="height:550px",
                                  h4("Add layer to the model"),
                                  selectInput("layer_name", label="Select layer type", choices=c("Dense", "Dropout"), width="100%"),
                                  
                                  conditionalPanel(
                                    condition = "input.layer_name == 'Dense'",
                                    p("Layer in which every neuron is connected to the every neuron in the previous layer"),
                                    p("For this layer number of neurons and activation function must be set."),
                                    br(),
                                    sliderInput("layer_units", label="Select number of neurons", min=1, max=128, value=16, width="100%"),
                                    selectInput("layer_activation", label="Select layer activation function", choices=c("linear", "relu", "elu"), width="100%")
                                  ),
                                  
                                  conditionalPanel(
                                    condition = "input.layer_name == 'Dropout'",
                                    p("Defines how much of layer neurons will be randomly dropped out during testing. Used to prevent overfitting."),
                                    p("For this layer a dropout rate (percentage of randomly dropped neurons) must be set."),
                                    br(),
                                    sliderInput("layer_dropout", label="Select dropout rate", min=0.01, max=0.99, value=0.5, width="100%"),
                                  ),
                                  
                                  actionButton('layer_add', 'Add layer', width="100%")
                                )
                              ),
                              
                              card(
                                h4("Current layers in the model"),
                                
                                div(tableOutput("current_layers"), style="max-height:331px;overflow:auto"),
                                br(),
                                actionButton('layer_clear', 'Clear layers', width="100%")
                              ),
                              col_widths = c(6,6)
                            )
                          ),
                          
                      ),
                      tabPanel("Train the model",
                               br(),
                               span(textOutput('compile_no_data'), style="color:red;padding:10px;display:block;text-align:center", width="100%"),
                               page_fluid(
                                 layout_columns(
                                   card(
                                     h4("Set Compile Parameters"),
                                     sliderInput("epochs", label="Select the number of epochs", min=1, max=50, value=5, width="100%"),
                                     selectInput("optimizer", label="Select optimizer function", choices=c("Adam", "Adamax", "SGD", "FTRL"), width="100%"),
                                     selectizeInput("learning_rate", label="Set the learning rate", choices=c(0.0001, 0.001, 0.01, 0.1), width = "100%"),
                                     actionButton('compile_train', 'Compile and train', width="100%")
                                   ),
                                   
                                   card(
                                     h4("Training status"),
                                     htmlOutput("train_status"),
                                     plotOutput("loss_training")
                                   ),
                                   col_widths = c("100%","100%")
                                 )
                               ),
                               card(
                                 h4("Model Summary"),
                                 verbatimTextOutput("model_summary"),
                                 br(),
                               )
                          ),
                      )
                     
                    ),
            tabPanel("Inspect Data", 
                      br(),
                      textOutput("no_upload"),
                      tableOutput("contents")),
            )
          )
    ),
)

# Define server logic required to draw a histogram
server <- function(input, output, session) {
    
    env <- reactiveValues(layers = data.frame(layer_type=character(),
                                              units=numeric(),
                                              act_func=character()),
                          history = NULL)
    df <- eventReactive(input$file, {
      read.csv(input$file$datapath, header=TRUE)
    })
    
    
    observeEvent(input$compile_train, ignoreInit=TRUE, {
      if (is.null(input$file)) {
        output$compile_no_data <- renderText({"UPLOAD DATA AND ADD LAYERS BEFORE TRAINING!"})
      }
      
      else if (nrow(isolate(env$layers)) < 1) {
        output$compile_no_data <- renderText({"ADD LAYERS BEFORE TRAINING!"})
      }
      else {
        data <- df()
        
        inp_shape = ncol(data) - 1
        out_shape = nrow(unique(data[ncol(data)])) + 1
        model <- NULL
        model <- tfmodel$new(name="playground_model",
                             input_shape=inp_shape,
                             output_shape=out_shape,
                             data=data,
                             test_split=0.2)
        
        
        model$add_layers_from_df(env$layers)
        model$build_model(input$optimizer, as.numeric(input$learning_rate))
        output$model_summary <- renderPrint(summary(model$model))
        
        output$train_status <- renderUI({HTML(paste0("Training the model for ", input$epochs, " epochs..."))})
        model$train(as.numeric(input$epochs))
        data_epochs = paste0("Model trained for ", input$epochs, " epochs")
        data_optimizer = paste0(input$optimizer, " optimizer with learning rate = ", input$learning_rate)
        output$train_status <- renderUI({HTML(paste(data_epochs, data_optimizer, sep="</br>"))})
        env$history <- model$history

        output$loss_training <- renderPlot(
          plot(model$history)
        )

        
      }
    })
    
    output$no_upload <- renderText("Upload the data first!")
    
    output$contents <- renderTable(
      {
        req(input$file)
        output$no_upload <- renderText("")

        return(head(df()))
      })
    
    observeEvent(c(input$layer_add), ignoreInit = TRUE, {
      if (is.null(input$file)) {
        output$layer_no_data <- renderText({"UPLOAD DATA BEFORE ADDING LAYERS!"})
      }
      else {
        output$layer_no_data <- renderText("")
      }
      req(input$file)
      if (input$layer_name == "Dense"){
        env$layers <- rbind(env$layers, data.frame(layer_type="Dense", units=input$layer_units, act_func=input$layer_activation))
      }
      else if (input$layer_name == "Dropout") {
        env$layers <- rbind(env$layers, data.frame(layer_type="Dropout", units=input$layer_dropout, act_func=""))
      }
    })
    
    observeEvent(input$layer_clear, ignoreInit = TRUE, {
      env$layers <- data.frame(layer_type=character(),
                               units=numeric(),
                               act_func=character())
    })
    
    observeEvent(c(input$layer_add), ignoreInit = TRUE, {
      req(input$file)
      output$n_layers_model <- renderText(nrow(env$layers))
      output$current_layers <- renderTable({
        df <- env$layers %>%
          rename("Layer Type"=layer_type,
                 "Units"=units,
                 "Activation Function"= act_func)
        return(df)
      })
    })
    
    

}

# Run the application 
shinyApp(ui = ui, server = server)
