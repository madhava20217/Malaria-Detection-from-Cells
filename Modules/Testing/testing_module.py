import os
import numpy as np
from time import gmtime, strftime
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score,average_precision_score, recall_score, f1_score, roc_curve ,auc, matthews_corrcoef, roc_auc_score, classification_report
from sklearn.metrics import r2_score, median_absolute_error, precision_recall_curve, mean_absolute_error, mean_squared_error, mean_squared_log_error
import seaborn as sns; sns.set(rc={"lines.linewidth":3})
from inspect import signature
from plot_metric.functions import BinaryClassification

class Evaluation():
    ## by default I am using model type as classification
    def __init__(self, actual, pred, threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], model_reference_name = 'sample_model', model_type = 'classification'):
        self.actual = actual
        self.pred = pred
        self.threshold = threshold
        self.model_reference_name = model_reference_name
        self.model_type = model_type
        
    def get_confusion_matrix_values(self, pred_value):
        tn, fp, fn, tp = confusion_matrix(self.actual, pred_value).ravel()
        return tn, fp, fn, tp

    def get_pred_value_threshold_lvl(self):
        pred_value = pd.DataFrame()
        for i in self.threshold:
            col_name = "Threshold_"+str(i)
            pred_value[col_name] = [1 if j>= i else 0 for j in self.pred]
        return(pred_value)
        
    def metrics_classification(self, pred_value):
        '''
        Calculating the metrics based on different sets of threshold:
        --------------------------------------------------------------
        metrics considered =  ['Threshold','TP','FP','FN','TN','Accuracy','Precision','recall','f1','mcc','roc_auc']
        '''
        key = ['Unique_ModelID','Model_Reference_name','Threshold','TP','FP','FN','TN','Accuracy','Precision0','Precision1','recall0','recall1','f1','mcc','roc_auc','Time_stamp']
        metrics_db = dict([(i, []) for i in key])        
        id = str(self.model_reference_name)+'_'+str(int(round(time.time() * 1000)))[:10]
        for i in self.threshold:
            metrics_db['Unique_ModelID'].append(id)
            col_name = "Threshold_"+str(i)
            metrics_db['Model_Reference_name'].append(self.model_reference_name)
            metrics_db['Threshold'].append(round(i,2))
            TN, FP, FN, TP = self.get_confusion_matrix_values(pred_value = pred_value[col_name])
            metrics_db['TP'].append(TP)
            metrics_db['FP'].append(FP)
            metrics_db['FN'].append(FN)
            metrics_db['TN'].append(TN)
            metrics_db['Accuracy'].append(accuracy_score(self.actual,pred_value[col_name]))
            metrics_db['Precision0'].append(precision_score(self.actual,pred_value[col_name],pos_label=0))
            metrics_db['Precision1'].append(precision_score(self.actual,pred_value[col_name],pos_label=1))
            metrics_db['recall0'].append(recall_score(self.actual,pred_value[col_name],pos_label=0))
            metrics_db['recall1'].append(recall_score(self.actual,pred_value[col_name],pos_label=1))
            metrics_db['f1'].append(f1_score(self.actual,pred_value[col_name]))
            metrics_db['mcc'].append(matthews_corrcoef(self.actual,pred_value[col_name]))
            metrics_db['roc_auc'].append(roc_auc_score(self.actual,pred_value[col_name]))
            metrics_db['Time_stamp'].append(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        return(pd.DataFrame(metrics_db))
    
    def metrics_regression(self):
        '''
        Calculating the below metrics for the regression model:
        -------------------------------------------------------
        - mean_absolute_error 
        - mean_squared_error 
        - mean_squared_log_error
        - median_absolute_error
        - r2_score          
        '''
        key = ['Unique_ModelID','Model_Reference_name','mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error','median_absolute_error', 'r2_score','Time_stamp']
        metrics_db = dict([(i, []) for i in key])

        metrics_db['Unique_ModelID'].append(str(self.model_reference_name)+'_'+str(int(round(time.time() * 1000)))[:10])
        metrics_db['Model_Reference_name'].append(self.model_reference_name)
        metrics_db['mean_absolute_error'].append(mean_absolute_error(self.actual,self.pred))
        metrics_db['mean_squared_error'].append(mean_squared_error(self.actual,self.pred))
        metrics_db['mean_squared_log_error'].append(mean_squared_log_error(self.actual,self.pred))
        metrics_db['median_absolute_error'].append(median_absolute_error(self.actual,self.pred))
        metrics_db['r2_score'].append(r2_score(self.actual,self.pred))
        metrics_db['mean_squared_error'].append(mean_squared_error(self.actual,self.pred))
        
    def metrics(self,pred_value = None):
        '''
        Calculating the metrics table for classification or regression problem.
        '''
        if self.model_type == 'classification':
            metrics_db = self.metrics_classification(pred_value = pred_value)
        elif self.model_type == 'regression':
            metrics_db = self.metrics_regression()
        return(metrics_db)


class evaluation_plots():
    
    def __init__(self,metrics_db,classification_metric = ['TP','FP','FN','TN','Accuracy','Precision0','Precision1','recall0','recall1','f1','mcc','roc_auc']):
        '''
        metric_db - Should be passed as a pandas Dataframe
        model_id - Should be passed as a list
        '''
        self.classification_metric = classification_metric
        self.metrics_db = metrics_db
        self.hue_feat = 'Unique_ModelID'
        
        
    def metric_plots_1(self):
        for i in self.classification_metric:
            sns.lineplot(x='Threshold', y=i,hue = self.hue_feat, markers=True, dashes=True, data=self.metrics_db)
            plt.show()

    def metric_plots_2(self,actual = None, pred=None, threshold = 0.5):
        if ((actual is not None) & (pred is not None)):                           
            bc = BinaryClassification(y_true = actual, y_pred = pred, labels=["Class 0", "Class 1"], threshold = threshold)
            plt.figure(figsize=(20,15))
            plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
            bc.plot_roc_curve(threshold = threshold)
            plt.subplot2grid((2,6), (0,2), colspan=2)
            bc.plot_precision_recall_curve(threshold = threshold)
            plt.subplot2grid((2,6), (0,4), colspan=2)
            bc.plot_class_distribution(threshold = threshold)
            plt.subplot2grid((2,6), (1,1), colspan=2)
            bc.plot_confusion_matrix(threshold = threshold)
            plt.subplot2grid((2,6), (1,3), colspan=2)
            bc.plot_confusion_matrix(threshold = threshold, normalize=True)
            plt.show()
            print(classification_report(actual, [0 if i<=threshold else 1 for i in pred]))

    def classification_report_chart(self, actual=None, pred=None, best_threshold = 0.5):
        print(classification_report(actual, [0 if i<=best_threshold else 1 for i in pred]))

    def roc_auc_plot(self, actual = None, pred = None):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.title('ROC CURVE')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
        label='AUC = %0.2f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.1,1.2])
        plt.ylim([-0.1,1.2])
        plt.ylabel('True_Positive_Rate')
        plt.xlabel('False_Positive_Rate')
        plt.show()

    def precision_recall_plot(self, actual = None, pred = None):
        precision, recall, _ = precision_recall_curve(actual, pred)
        average_precision = average_precision_score(actual, pred)
        step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        plt.step(recall, precision, color='b', alpha=0.2,where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: Avg Precision ={0:0.2f}'.format(average_precision))

    def confusion_matrix_plot(self,actual = None, pred = None, best_threshold = 0.5):
        if ((actual is not None) & (pred is not None)):
            pred_value = [1 if i >= best_threshold else 0 for i in pred]
            cm = confusion_matrix(actual, pred_value)
            plt.clf()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.gray_r)
            classNames = ['Negative','Positive']
            plt_title = "Confusion Matrix plot - Threshold ("+str(best_threshold)+")"
            plt.title(plt_title)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            tick_marks = np.arange(len(classNames))
            plt.xticks(tick_marks, classNames, rotation=45)
            plt.yticks(tick_marks, classNames)
            s = [['TN','FP'], ['FN', 'TP']]
            for i in range(2):
                for j in range(2):
                    plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]),bbox=dict(facecolor='white', alpha=0.5))
            plt.show()


class ModelEvaluation(Evaluation,evaluation_plots):

    def __init__(self, actual, pred, threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], model_reference_name = 'sample_model', model_type = 'classification',
        plot_classification_metric = ['TP','FP','FN','TN','Accuracy','Precision0','Precision1','recall0','recall1','f1','mcc','roc_auc'],maximising_metrics='f1'):
        self.actual = actual
        self.pred = pred
        self.threshold = threshold
        self.model_reference_name = model_reference_name
        self.model_type = model_type
        self.plot_classification_metric = plot_classification_metric
        self.maximising_metrics = maximising_metrics

    def evaluate(self,evaluate_save=False, plots_show = True, model_class=0, bins=10):
        evalu = Evaluation(actual = self.actual, pred = self.pred,threshold = self.threshold,model_reference_name=self.model_reference_name,model_type = self.model_type)
        pred_value = evalu.get_pred_value_threshold_lvl()
        metrics_db = evalu.metrics(pred_value)
        if evaluate_save:
            writer = pd.ExcelWriter('evaluation_results.xlsx', engine='xlsxwriter')
            metrics_db.to_excel(writer, sheet_name='Metrics_details')
            writer.save()
            print("The results are save to - ",os.getcwd()+'\\evaluation_results.xlsx')
        if plots_show:
            eval_plt = evaluation_plots(metrics_db = metrics_db, classification_metric = self.plot_classification_metric)
            eval_plt.metric_plots_1()
            best_threshold = metrics_db[metrics_db[self.maximising_metrics] == max(metrics_db[self.maximising_metrics])]['Threshold'].reset_index(drop=True)[0]
            eval_plt.classification_report_chart(actual = self.actual, pred=self.pred, best_threshold = best_threshold)
            eval_plt.confusion_matrix_plot(actual = self.actual, pred=self.pred, best_threshold = best_threshold)
            eval_plt.roc_auc_plot(actual = self.actual, pred=self.pred)
            eval_plt.precision_recall_plot(actual = self.actual, pred=self.pred)
            plt.close()
        return(metrics_db, best_threshold, self.maximising_metrics)

    def Compare_models(self, evaluate_db = None ,model_id = None, comparison_metrics = None):
        if comparison_metrics:
            _metric = comparison_metrics
        else:
            _metric = self.plot_classification_metric
        if model_id:
            data = evaluate_db[evaluate_db['Unique_ModelID'].isin(model_id)]
            eval_plt = evaluation_plots(metrics_db = data, classification_metric = _metric)
            eval_plt.metric_plots()
            plt.close()