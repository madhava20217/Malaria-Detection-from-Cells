# About the Testing Module

This module is capable of taking any model wether classifier or regressor and give metrics and plots various graphs.

### Classes present in the Module:

- class Evaluation
- class evaluation_plots
- class ModelEvaluation

We only need to run class Evaluation to get plots and metrics, if one needs to get specifically plots or metrics then one can run respective classes separetely as well.

### How to Use ?

To import this module follow this command if you are in the root folder of the project.

```
import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"\\Modules\\Testing")
import testing_module
```

#### class ModelEvaluation

This class can display metrics as well as show plots at the same time 

- Paramas while calling constructor: 
    - required:
        -   actual = actual value in numpy array
        -   pred = predicted value in numpy array
    -   optional:
        -   model reference name = name of your model in string
        -   model_type = based on your model 'classification' or 'regression', 
        by default = 'classification'
        - plot_classification_metric = ['TP','FP','FN','TN','Accuracy','Precision0','Precision1','recall0','recall1','f1','mcc','roc_auc'] by default
        -  maximising_metrics='f1' by default

Example:
```
ModelEvaluation = testing_module.ModelEvaluation(train_y,y_pred_train ,model_reference_name = 'Naive Bayes',model_type = 'classification') # if classification
```

- Methods in this class:
    - evaluate()
        - evaluate_save = False by default
        if passed True then it will create an excel file (in .xlsx format in the same directory) saving metrics in the excel file.
        - plots_show = True by default
        if passed False then it won't show any plots(details of plots are mentioned in class evaluation_plots in README)
        - model_class = 0 by default
        - bins = 10 by default (no need to change this)
    - Compare_models()
        - evaluate_db = None by default
        You can pass a db array consisting of models metrics
        - model_id =  None by default
        Index of model reqd for comparision in evaluate_db
        - comparison_metrics = None by default
        If left none then it will take metrics of the object used else one can also pass some other metrics as well

Example:
```
ModelEvaluation.evaluate(evaluate_save=True,plots_show=True)
```

#### class Evaluation

This class display metrics only.

Calculating the metrics based on different sets of threshold for classification model:

['Threshold','TP','FP','FN','TN','Accuracy','Precision','recall','f1','mcc','roc_auc']

Calculating the below metrics for the regression model:
- mean_absolute_error 
- mean_squared_error 
- mean_squared_log_error
- median_absolute_error
- r2_score  
       
##### To use metrics in this class follow steps:
-  First create the class by calling the constructor:
    - Paramas while calling constructor: 
        - required:
            -   actual = actual value in numpy array
            -   pred = predicted value in numpy array
        -   optional:
            -   model reference name = name of your model in string
            -   model_type = based on your model 'classification' or 'regression', 
            by default = 'classification'
    - Example:
        ```
        ModelEvaluation = testing_module.Evaluation(train_y,y_pred_train,model_reference_name = 'Naive Bayes',model_type = 'classification')
        ```
- [Follow this step only if its classification] Now we need db of y_pred (lets say pred_value) so we need to call get_pred_value_threshold_lvl()
    - Params of  get_pred_value_threshold_lvl()
        - None
    - Return value 
        - pred_value
    - Example:
        ```
         pred_y = ModelEvaluation.get_pred_value_threshold_lvl()
        ```

- Now to finally get metrics results use metrics()
    - Params of metrics()
        - y_pred : None by default 
        If your model performs regression then nothing needed to pass 
        Else pass pred_y got from previous step
    - Return Value of metrics()
        - metrics in db format
    - Example:
         ```
         ModelEvaluation.metrics(pred_y) # if classification
         ModelEvaluation.metrics() # if regression
        ```

#### class evaluation_plots

This class displays following plots only:

- lineplot 
- plot_roc_curve
- precision_recall_plot
- plot_precision_recall_curve
- plot_class_distribution
- plot_confusion_matrix

To use this class:
- call metric_plots_1() 
    - No params
    - Plots lineplot
- call metric_plots_2()
    - Params:
        - actual = None by default
        if regression keep it None
        else for classification put actual value in numpy array
        - pred = None by default
        if regression keep it None
        else for classification put actual value in numpy array
        - threshold = 0.5 by default
    - Plots 
        - plot_roc_curve
        - precision_recall_plot
        - plot_precision_recall_curve
        - plot_class_distribution
        - plot_confusion_matrix
    - Prints classification report if classification type model