#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 22:18:06 2022

@author: apple
"""
import LSTM
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import pandas as pd

    
def graph(df_test,df_train,days):
    data_all = pd.concat([df_test, df_train], sort=False)
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(13)
    data_all.rename(columns = {'Y':'Actuals'}, inplace = True)
    data_all.rename(columns = {'demand_Prediction':'Forecast'}, inplace = True)
    data_all[['Actuals','Forecast']].plot(ax=ax,style=['-','.'])
    ax.set_xbound(lower=df_test.index.max()-pd.Timedelta(days, unit='d'), upper=df_test.index.max())
    plt.suptitle(str(days) + ' Days Forecast vs Actuals')
    return f

def graphbar(df_test,df_train,days):
    data_all = pd.concat([df_test, df_train], sort=False)
    data_all.index = data_all.index.date
    f = data_all.head(days).plot.bar(color=['blue', 'black'])
    plt.suptitle(str(days) + ' Days Forecast vs Actuals')
    return f
    
def LoadLSTM():
    df = pd.read_csv('dataset.csv')
    df = LSTM.PreData(df)
    df, X_train, y_train, df_test, df_train, sc = LSTM.ScaleData(df)
    history,model = LSTM.FitLSTM(X_train,y_train)
    #LSTM.LossPlot(history)
    df_test =LSTM.PredictDemand(df_test,sc,model)
    return [df_test,df_train]

def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)


class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)


class BaseGUI(object):
    
    def __init__(self):
        # theme color
        sg.ChangeLookAndFeel('BrownBlue')
        
        self.FONT = (16)
        #self.SIZE = (400, 100)
        
        # Left
        input_column = [
            [sg.Text('Predict Days', font=self.FONT, size=(30, 1))],
            [sg.InputText(key='_NUM_', size=(32, 1))],
            [sg.Text('Choose Chart',size=(30, 1), font=self.FONT,justification='left')],
            [sg.Combo(['Line Chart'],default_value='Line Chart',key='_CHART_',size=(30, 1))],            
            [sg.Btn('OK', key='_SUMMIT_', font=(16), size=(10, 1)),sg.Exit(font=(16), size=(10, 1))],     
        ]
        
        # Right
        output_column = [
            [sg.Output(size=(100, 10),font=(0),background_color='light gray',key = '_OUTPUT_')],
        ]

        self.layout = [
            [sg.Image(filename=r"France-Map-Region-PNG-Pic.png",size=(1000,200), pad=(0, 0))],
            [sg.Column(input_column),
             sg.VSeperator(),
             sg.Column(output_column),],
            [sg.Canvas(key='controls_cv')],
            [sg.Canvas(key='_CANVAS_',size=(800,200), pad=(0, 0))],    
        ]

        self.window = sg.Window('France Electricity Demand Forecast', layout=self.layout, finalize=True)


    def run(self):
        while True:
            event, value = self.window.Read()
            if event == '_SUMMIT_':
                val = value["_NUM_"]
                chart = value["_CHART_"]
                if val == '':
                    self.window.Element("_OUTPUT_").Update("Predict Days can't be blank, please enter days")
                    
                else:
                    try:
                        days = int(val)
                        self.window.Element("_OUTPUT_").Update("Start forecating")
                        df_test,df_train = LoadLSTM()
                        if chart == "Line Chart":
                           fig = graph(df_test,df_train,days)
                        elif chart == "Bar Chart":
                           fig = graphbar(df_test,df_train,days)
                        
                        # add the plot to the window
                        draw_figure_w_toolbar(self.window['_CANVAS_'].TKCanvas, fig, self.window['controls_cv'].TKCanvas)
                        MSE, MAE, RMSE, R2_SCORE = LSTM.Score(df_test)
                        result = ("LSTM Completed\n" + '\nR2_SCORE : %f' % R2_SCORE
                                                              + '\nMSE : %f' % MSE
                                                              + '\nMAE : %f' % MAE
                                                              + '\nRMSE : %f' % RMSE)
                        self.window.Element("_OUTPUT_").Update(result)

                    except:
                        self.window.Element("_OUTPUT_").Update("Please enter correct days")
             
            elif event in (sg.WIN_CLOSED, 'Exit'):
               break
    
            elif event is None:
                break
        self.window.close()

if __name__ == '__main__':
	tablegui = BaseGUI()
	tablegui.run()

