import yaml #enable to process configuration.yaml file.
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show
from bokeh.transform import jitter
from bokeh.palettes import HighContrast3
from bokeh.transform import factor_cmap, factor_mark
from bokeh.plotting import figure


def open_config_yaml(directory):
    '''
    Input: 1. directory - config.yaml file

    Function return a dictionary from directory file

    Output: dictionary from (config file)
    '''
    with open(directory, 'r') as config_info:
        config_dict = yaml.safe_load(config_info)
    return config_dict


def compare(elem, plus, minus):
    '''
    Input: 1. elem - element for compare (number)
           2. plus - upper limit of comparison
           3. minus - low limit of comparison

    Function compare result BT with limits and return result

    Output: status for this observation. 1 - infected, 
                                         0 - healthy, 
                                         0.5 - insufficient (zone between limits)
    '''
    if elem >= plus:
        return 1
    elif elem <= minus:
        return 0
    elif np.isnan(elem):
        return np.nan
    else:
        return 0.5
    

def compare_results(row, list_e, how='status'):
    '''
    Input: 1. row - row from df
           2. list_e - list names(columns) in row for compare
           3. how - method for compare:
                                'status' - compare labels(1, 0.5, 0), return max
                                'amount_0' - compare amount bacil forms in %, compare 
                                            max with cut-off = 0, return '+' if greater
                                'amount_10' - compare amount bacil forms in %, compare 
                                            max with cut-off = 10, return '+' if greater

    Function compare results Histology between same places (i.e. body) and return maximum label.
        In case 'amount' also compare with cut-off. Return '+' only if max is greater than the cut-off.

    Output: status for this observation. 1 or '+' - infected, 
                                         0 or '-' - healthy, 
                                         0.5 - detected small amount of HP
    '''
    if how == 'status':
        comp_list = [1, 0.5]
        final_result = 0.0
        for e in list_e:
            if row[e] >= comp_list[0]:
                return 1.0
            elif row[e] == comp_list[1]:
                final_result = 0.5
    
    elif how == 'amount_0':
        comp_list = 0.0
        final_result = '-'
        for e in list_e:
            if row[e] > comp_list:
                return '+'
    
    elif how == 'amount_10':
        comp_list = 10.0
        final_result = '-'
        for e in list_e:
            if row[e] > comp_list:
                return '+'
            
    return final_result


def compare_bacil(row, list_e, total='no'):
    '''
    Input: 1. row - row from df
           2. list_e - list names(columns) in row for compare
           3. total - method for compare:
                                'no' - compare 2 columns, return maximum value
                                'yes' - compare all values in row, return maximum 

    Function compare results Histology between two columns for same places (i.e. body) and return 
        maximum bacil value in %. In case total = 'yes' compare all values in row and return maximum.

    Output: maximum value of bacil forms in %
    '''

    if total == 'no':
        if row[list_e[0]] >= row[list_e[1]]:
            return row[list_e[0]]
        else:
            return row[list_e[1]]
    elif total == 'yes':
        dis = sorted(list(row), reverse=True)
        return dis[0]




def create_result(df, list_compile, value='status', bacil='no'):
    '''
    Input: 1. df - dataframe 
           2. list_compile - list of common words to search for similar areas in df
           3. value - see function 'compare_results', arg 'how'
           4. bacil - which subfunction we use:
                                'no' - use 'compare_results' and receive labels
                                'yes' - use 'compare_bacil' and receive % bacil forms

    Function compare same columns in df and create a new columns with overall results (labels or % bacil forms).
    The same columns are identified based on the common words passed in the 'list_compile' 
                                                (in this case, stomach sections were used)

    Output: df with additional data (add columns)
    '''
    columns = list(df.columns)
    for name in list_compile:
        elem_list = []
        for elem in columns:
            #using re give us match list
            result = re.search(name, elem)
            if result:
                elem_list.append(elem)

        if bacil == 'no':
            new_col_name = name + '_status'
            df[new_col_name] = df.apply(lambda x: compare_results(x, elem_list, how=value), axis=1)
        
        elif bacil == 'yes':
            new_col_name = name + '_bacil'
            #function 'compare_bacil' needs list with 2 args. 
            if len(elem_list) < 2:
                elem_list.append(elem_list[0])   #If we have only 1 arg, then copy it and function just return this column 
            df[new_col_name] = df.apply(lambda x: compare_bacil(x, elem_list), axis=1)            
    
    if bacil == 'yes':
        df['total_bacil'] = df.apply(lambda x: compare_bacil(x, elem_list, total='yes'), axis=1)  
    return df

'''
def total_result(row, e_list):
    result = 0.0
    for e in e_list:
        if row[e] == 1:
            return 1.0
        elif row[e] == 0.5:
            result = 0.5        
    return result
'''

def create_dict(df, list_compile):
    '''
    Input: 1. df - dataframe
           2. list_compile - list of common words to search for similar areas in df

    Function create and return a dictionary for panel with plots.

    Output: dict like [key = 'elem from list_compile'] = [value = [column with % bacil(related with name), 
                                    label(+ or -), BT (Heliforce)], [same columns, but with BT (Helicarb)]]
    '''
    columns = list(df.columns)
    dict_names = {}
    for name in list_compile:
        elem_list = []

        #create list of column names related with 'name'
        for elem in columns:
            result = re.match(name, elem)
            if result:
                elem_list.append(elem)

        elem_list.append('13С BT (Heliforce), Δ')
        elem_list1 = elem_list.copy()

        #change last element in list
        elem_list.insert(-1, '13С BT (Helicarb), Δ')
        elem_list2 = elem_list[0:-1]
        #add new pair in dictionary
        dict_names[name] = [elem_list1, elem_list2]
    return dict_names



def create_heatmap(df):
    """
    Input: 1. df - dataframe to heatmap. All categorical columns should be previously convert into numerical !!!

    Easy function to create a nice heatmap. 

    Output: Heatmap
    """
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')  
    plt.show()


def render_bt_plot(df):
    """
    Input: 1. df - dataframe to plot. Should contain numerical columns (not categorical!)

    Function create a plot for Breath Tests. Compare positive and negative results for two different loads

    Output: Plot
    """
    p = figure(title='Dependancy between BT', y_axis_label=f'R.Helicarb', x_axis_label=f'Heliforce', toolbar_location=None)
    source = ColumnDataSource(df)
    
    p.circle(x='13С BT (Heliforce), status', 
              y=jitter('13С BT (Helicarb), status', width=0.2),
              source=source, 
              fill_alpha=0.6, 
              size=10)
    show(p)


def render_plot(dict_, df, key='total', bt_load='Heliforce'):
    """
    Input: 1. dict_ - dictionary that we return from 'create_dict' function
           2. df - dataframe to plot. 
           3. key - keyword for dict_. Choice a set of values
           4. bt_load - keyword for selecting a list with a specific load

    Function create a plot for BT value depending on % of bacil forms (with labels)

    Output: Plot
    """
    #load selection
    if bt_load == 'Heliforce':
        var = 0
        i = 4.4
    else:
        var = 1
        i = 4.5

    x = dict_[key][var][0]
    y = dict_[key][var][2]

    title = 'Dependence of the BT result from the number of bacillary forms(in %)'
    p = figure(title=title, y_axis_label=f'{y}, ‰', x_axis_label=f'{x}, %', toolbar_location=None)
    
    #Creating a ColumnDataSource from the DataFrame
    source = ColumnDataSource(df.loc[:,dict_[key][var]])

    #Create a list of labels from df
    status_of_patients = ['+', '-']    
    
    #Type of figures on the plots
    markers = ['hex', 'triangle']
    
    #Adding scatter plot with 'status_of_patients' as hue and colormap
    p.scatter(x=x, 
              y=y, 
              source=source, 
              fill_alpha=0.3, 
              size=5, 
              color=factor_cmap(dict_[key][var][1], 'HighContrast3', status_of_patients),  
              #'Label' - Which column we use to split for different types
              #'HighContrast3' - which colour we use for each element (also we can use [] of colours)
              #status_of_patients - different values that we search in 'Label' column
              
              marker=factor_mark(dict_[key][var][1], markers, status_of_patients), 
              #The same as 'color'
              legend_field=dict_[key][var][1])
    
    #Add a cut-off line
    p.line([0, 100], i, legend_label='cut-off value', line_width=2)

    # Displaying the plot with legend
    p.legend.location = 'top_right'
    p.legend.title = "Patient's status"
    
    return p
