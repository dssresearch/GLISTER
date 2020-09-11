'''
    Author : Ayush Dobhal
    Date created : 6/14/2020
    Description : This file contains code for plotting plot.ly interactive plots.
    Define a nested dict and call the functions accordingly as given in examples.
'''
import plotly.graph_objects as go
import plotly.offline
from collections import defaultdict


def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))
		
def bar_plot(Y_axis, X_axis, plot_title, X_title, Y_title, filename="file.html"):
    plots = X_axis
    ya = Y_axis
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plots,
        y=ya,
        name='Subset Selection',
        marker_color='#483D8B',
        text=ya
    ))
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_layout(title=plot_title,
                      xaxis_title=X_title,
                      yaxis_title=Y_title)
    # fig.update_layout(barmode='group', xaxis_tickangle=0)
    plotly.offline.plot(fig, filename=filename)
	
def create_plot(cifar, X_axis, plot_title, X_title, Y_title, filename="file.html"):
    fig = go.Figure()
    color_idx = 0
    for model, dict1 in cifar.items():
        for label, dict2 in dict1.items():
            plot_color = DEFAULT_PLOTLY_COLORS[color_idx]
            color_idx += 1
            final_label = label
            dash_value = None
            if "Random" in final_label:
                dash_value = 'dot'
            #elif "trn" in final_label:
            #    dash_value = 'dash'
            fig.add_trace(go.Scatter(x=X_axis, y=dict2["Y1"], mode='lines+markers', marker=dict(
                color=(plot_color)
            ), line=dict(color=plot_color, dash=dash_value), name=final_label))
            #fig.add_trace(go.Scatter(x=X, y=dict2["Y2"], mode='lines+markers', marker=dict(
            #    color=(plot_color),
            #), line = dict(color='gray', width=2, dash='dash'), name=final_label))

    fig.update_layout(title=plot_title,
                      xaxis=dict(
                          tickmode='array',
                          tickvals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1]
                      ),
                       xaxis_title=X_title,
                       yaxis_title=Y_title)
    plotly.offline.plot(fig, filename=filename)
		
budgets_X = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1]

cifar = nested_dict(3, list)

h1 = [2, 1, 0.5, 0.25, 0.05, 0.01]

# CIFAR10
# KNN Submod
hyper10 = nested_dict(3, list)
hyper10["10"]["KNNSubmod Budget=10% with momentum=0.9"]["Y1"] = [66.24, 80.32, 81.99, 82.13, 80.59, 78.57]
hyper10["10"]["KNNSubmod Budget=10%  with momentum=0"]["Y1"] = [81.78, 80.11, 80.31, 80.41, 79.01, 72.46]
hyper10["10"]["Random Selection Budget=10% with momentum=0.9"]["Y1"] = [26.44, 80.74, 82.26, 81.65, 80.26, 78.35]
hyper10["10"]["Random Selection Budget=10%  with momentum=0"]["Y1"] = [80.29, 79.86, 80.38, 78.91, 77.64, 72.44]
hyper10["25"]["KNNSubmod Budget=25% with momentum=0.9"]["Y1"] = [27.79, 86.98, 88.17, 89.27, 88.79, 87.31]
hyper10["25"]["KNNSubmod Budget=25% with momentum=0"]["Y1"] = [89.19, 89.23, 88.74, 88.53, 86.74, 83.38]
hyper10["25"]["Random Selection Budget=25% with momentum=0.9"]["Y1"] = [56.99, 86.23, 87.57, 88.82, 88.05, 86.74]
hyper10["25"]["Random Selection Budget=25% with momentum=0"]["Y1"] = [88.4, 88.95, 88.09, 86.71, 86.25, 82.07]
hyper10["100"]["Full dataset with momentum=0.9"]["Y1"] = [10, 91.31, 93.5, 94.59, 94.71, 94.26]
hyper10["100"]["Full dataset with momentum=0"]["Y1"] = [94.92, 95.02, 94.8, 94.76, 93.82, 92.29]

DEFAULT_PLOTLY_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

create_plot(hyper10, h1, 'Resnet18 on CIFAR10 Dataset Hyperparameter Selection for budget=10%, 25%, 100%', 'Learning Rate', 'Accuracy', 'HP_Resnet18_KNNSubmodCIFAR10_all_plots.html')

# Comparison bar plots
#mnist 25%sub
plots_c6 =['KNNSubmod LR=0.05, mom=0.9', 'NBSubmod LR=0.5, mom=0.9', 'Random LR=0.25, mom=0.9', 'Full dataset LR=0.01,Mom=0.9',
           'KNNSubmod LR=2, mom=0', 'NBSubmod LR=2, mom=0', 'Random LR=0.25, mom=0', 'Full dataset LR=0.05,Mom=0']
vals_c6 = [99.64, 99.64, 99.57, 99.66,
           99.57, 99.57, 99.64, 99.66]

bar_plot(vals_c6, plots_c6, "MNIST Hyperparamters selection Budget=25%", "Selection Parameters", "Accuracy", "mnist_25sub_bar_plot_hyper.html")
