import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import date, datetime
from tqdm import tqdm
from pyalex import Works, Authors, Sources, Institutions, Topics, Publishers, Funders
from AJNR_citation import get_author_position_metrics, add_dummified_column, extract_mesh_and_topics, remove_outliers_iqr, plot_regressions

if __name__ == '__main__':
    #pip install pyalex
    file = '/home/hvv/Documents/reviews/AJNR/publication_stats/AJNR_citation_reg_inp_mw.xlsx'
    dir_out = os.path.dirname(file)
    data = pd.read_excel(file)#.rename(columns=cols)

    cat_vars = [c for c in data.columns if c.startswith('mesh_') or c.startswith('topic_')]
    cat_vars.extend([c for c in data if 'cat_' in c])
    cat_vars_select = [c for c in cat_vars if data[c].sum()>20]
    cat_vars_select.extend([ 'Brain', 'Head and Neck', 'Spine', 'Pediatrics', 'Nuclear Medicine', 'AI'])

    continuous_vars = [data.columns[i] for i in range(len(data.columns)) if
                       data.dtypes[i] not in [str, list, tuple] and data.columns[i] not in cat_vars]
    continuous_vars_select = [c for c in continuous_vars if not (('citations' in c or 'fwci' in c) and not 'author' in c) and not c in cat_vars_select] #exclude citation counts and fwci from continuous vars to plot (as they are targets)
    vartypes = pd.DataFrame([continuous_vars_select, cat_vars_select], index=['continuous', 'categorical']).T

    target_vars = ["total_citations","fwci"]

    vartype_dct = {}
    for c in data.columns:
        if c in continuous_vars_select:
            vartype_dct[c] = 'continuous'
        elif c in cat_vars_select:
            vartype_dct[c] = 'categorical'


    data = remove_outliers_iqr(data, continuous_vars_select, k=3)
    plot_regressions(data,
                    xvars=list(vartype_dct.keys()),
                    targets=target_vars,
                    vartypes=vartype_dct,
                    dir_fig=os.path.join(dir_out,'figures_mw'),
                    plot_formula=True)


    file_out = '/home/hvv/Documents/reviews/AJNR/publication_stats/AJNR_citation_reg_inp_mw2.xlsx'
    vartypes.to_excel(file_out)
    print(1)