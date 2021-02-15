############### PROJET 2 #################

import pandas as pd
import numpy as np 
import random
from collections import Counter 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

##########################################
#                                        #
# Fonctions de calcul et de descriptions #
#                                        #
##########################################

def getMissingValuesPercentPer(data, column):
    ''' 
        Calculates the mean percentage of missing values
        in a given pandas dataframe per unique value 
        of a given column
        
        Parameters
        ----------------
        data   : pandas dataframe
                 The dataframe to be analyzed
                 
        column : str
                 The column in data to be analyzed
        
        Returns
        ---------------
        A pandas dataframe containing:
            - a column "column"
            - a column "Percent Missing" containing the percentage of 
              missing value for each value of column     
    '''
    
    percent_missing = data.isnull().sum(axis=1) * 100 / len(data.columns)
    
    return pd.DataFrame({column: data[column], 'Percent Missing': percent_missing})\
                        .groupby([column])\
                        .agg('mean')


#------------------------------------------

def descriptionJeuDeDonnees(sourceFiles):
    ''' 
        Outputs a presentation pandas dataframe for the dataset.
        
        Parameters
        ----------------
        A dictionary with:
        - keys : the names of the files
        - values : a list containing two values : 
            - the dataframe for the data
            - a brief description of the file
        
        Returns
        ---------------
        A pandas dataframe containing:
            - a column "Nom du fichier" : the name of the file
            - a column "Nb de lignes"   : the number of rows per file 
            - a column "Nb de colonnes" : the number of columns per file
            - a column "Description"    : a brief description of the file     
    '''

    print("Les données se décomposent en {} fichiers: \n".format(len(sourceFiles)))

    filenames = []
    files_nb_lines = []
    files_nb_columns = []
    files_descriptions = []

    for filename, file_data in sourceFiles.items():
        filenames.append(filename)
        files_nb_lines.append(len(file_data[0]))
        files_nb_columns.append(len(file_data[0].columns))
        files_descriptions.append(file_data[1])

        
    # Create a dataframe for presentation purposes
    presentation_df = pd.DataFrame({'Nom du fichier':filenames, 
                                    'Nb de lignes':files_nb_lines, 
                                    'Nb de colonnes':files_nb_columns, 
                                    'Description': files_descriptions})

    presentation_df.index += 1

    
    return presentation_df

#------------------------------------------

def addCompositeRatioIndicators(list_names_nums_denoms, columns_with_update, 
                                groupbyCriteria, data):
    ''' 
        Adds ratio indicators in % to a pandas dataframe
        
        Parameters
        ----------------
        list_names_nums_denoms : {(new indicator nameA, numeratorA, denominatorA), 
                                  (new indicator nameB, numeratorB, denominatorB)}
                                 a dictionary of tuples containing the name, numerator 
                                 and denominator for each new ratio indicator
                                 
        columns_with_update.   : [{column name x : information to add for A,
                                   column name y : information to add for A},
                                  {column name x : information to add for B,
                                   column name z : information to add for B}]
                                 list of dictionaries containing column name and the information 
                                 that needs to be added regarding the new indicators
                                 
        groupbyCriteria        : str
                                 the grouping criteria
                                  
        data                   : pandas dataframe 
                                 used to calculate and add the new ratio indicator

        
        Returns
        ---------------
        The pandas dataframe given in parameter with an added row for each new indicator   
    '''
    
    # Grouping data by criteria
    data_per_criteria = {k: v for k, v in data.groupby(groupbyCriteria)}
    
    data_info_per_criteria = {}

    for criteria, criteria_df in data_per_criteria.items():
        data_per_criteria[criteria] = criteria_df.iloc[:,3:23].set_index('Indicator Code')
        data_info_per_criteria[criteria] = pd.concat([criteria_df.iloc[:,0:4],
                                                      criteria_df.iloc[:,23:]], axis=1)\
                                                    .set_index(['Indicator Code'])
        
    # Adding infos for the new indicators
    for criteria, criteria_info_df in data_info_per_criteria.items(): 
        
        new_info_rows = []
        
        for i, (name, num, denom) in enumerate(list_names_nums_denoms):
            
            new_info_row = criteria_info_df.xs('SE.SEC.ENRL').copy()
            new_info_row.name = name
            
            for col_name, val in columns_with_update[i].items():
                new_info_row[col_name] = val
                
            new_info_rows.append(new_info_row)
            
        data_info_per_criteria[criteria] = criteria_info_df.append(new_info_rows)
        
    # Calculating the values for the new indicators
    for criteria, criteria_df in data_per_criteria.items():   
        
        new_rows = []
            
        for name, num, denom in list_names_nums_denoms:
            new_row = (criteria_df.loc[num] / criteria_df.loc[denom]) * 100
            new_row.name = name
            
            new_rows.append(new_row)
            
        data_per_criteria[criteria] = criteria_df.append(new_rows)
        
    # Grouping the data back into a single dataframe
    data = pd.DataFrame()

    for criteria, criteria_df in data_per_criteria.items():
        info_df = data_info_per_criteria[criteria]\
                  .reset_index()\
                  .rename(columns={'index':'Indicator Code'})

        criteria_df = criteria_df.reset_index(drop=True)

        criteria_data = pd.concat([info_df, criteria_df], axis=1)

        data = pd.concat([data, criteria_data])

    return data

#------------------------------------------

def getStatsFor(data, criterion):
    ''' 
        Calculate the median, the mean and the standard deviation for each mean
        value of the critierion column.
        
        Parameters
        ----------------
        data      : pandas dataframe
                    The data for each value of the criterion column.
        
        criterion : str
                    The column name to calculate the statistics for.
                                  
        
        Returns
        ---------------
        data_per_criterion : pandas dataframe
                             For presentation of the calculated statistics
    '''
    
    data_per_criterion = {}
    data_criterion_totale = pd.DataFrame()
    
    # Group data by the given criterion
    for criterion_idx, criterion_df in data.groupby(criterion):         
        criterion_df = criterion_df.drop(columns=['Country Name', 'Country Code', 'Indicator Name', 
                                            'Region', 'Topic', 'Indicator Description', 'Long definition'])
        
        # Calculate the mean on a given indicator for the region, all countries included, per year
        tmp_data_df = criterion_df.groupby(['Indicator Code']).agg('mean').T
        
        # Calculate the median, mean, std for a given indicator for the region, on all years
        median = tmp_data_df.median()
        mean = tmp_data_df.mean()
        std = tmp_data_df.std()

        # Create a dataframe to hold the data for a given criterion
        # and store it inside a dictionary with the criterion as key
        data_per_criterion[criterion_idx] = pd.concat([median, mean, std], axis=1,
                                              keys=['median', 'mean', 'standard deviation'])

        data_per_criterion[criterion_idx].columns = pd.MultiIndex.from_product([[criterion_idx], 
                                                                      ['median', 'mean', 'standard deviation']],
                                                                     names=[criterion, 'grandeur stat'])
        
    # Create a final presentation dataframe containing all the stats
    for criterion_idx, criterion_df in data_per_criterion.items():
        data_criterion_totale = pd.concat([data_criterion_totale, criterion_df.T])
            
    return data_criterion_totale

#------------------------------------------

def getMedianOfTheAbsDifferenceFor(data, criterion, ref_criterion):
    ''' 
        Ranks the given values of a criterion in terms of least 
        absolute mean difference in median values compared to that 
        of a reference criterion.
        
        Parameters
        ----------------
        data          : pandas dataframe
                        The data for each value of the criterion column and
                        the reference criterion.
        
        criterion     : str
                        The column name to calculate the statistics for.
        
        ref_criterion : str
                        The value of criterion to serve as reference.
                                  
        
        Returns
        ---------------
        ranked_df : pandas dataframe
                    For presentation of the ranked values
    '''
    
    # Calculate the mean, median, standard deviation for each value 
    # of the criterion
    criterion_stats = getStatsFor(data, criterion)
    
    # Keep only the median
    criterion_stats.drop(index=['mean','standard deviation'], level=1, inplace=True)

    # Isolate median value for ref_criterion
    row_ref_criterion = criterion_stats.loc[ref_criterion,:]

    # Calculate for each value of the criterion the difference in %
    # between the median value for the criterion and the reference
    # criterion
    criterion_stats = (criterion_stats.sub(row_ref_criterion).abs().div(row_ref_criterion)*100)

    # Remove the ref_criterion row from dataframe
    criterion_stats.drop(index=[ref_criterion], inplace=True)

    # Create new presentation dataframe
    ranked_df = pd.DataFrame()

    # Calculate the mean difference on all indicators
    #ranked_df['Écart Moyen Tous Indicateurs (%)'] = criterion_stats.mean(axis=1)
    ranked_df['Médianes Écarts Tous Indicateurs (%)'] = criterion_stats.median(axis=1)


    # Assemble for presentation
    ranked_df = ranked_df.sort_values(['Médianes Écarts Tous Indicateurs (%)'])\
                         .reset_index()\
                         .drop(columns=['grandeur stat'])

    # Add Rank column and set it as index
    ranked_df['RANK']=ranked_df['Médianes Écarts Tous Indicateurs (%)'].rank()
    ranked_df.set_index('RANK', inplace=True)
    
    return ranked_df

#------------------------------------------

def getMedianOfTheNoAbsDifferenceFor(data, criterion, ref_criterion):
    ''' 
        Ranks the given values of a criterion in terms of least 
        mean difference in median values compared to that of a 
        reference criterion.
        
        Parameters
        ----------------
        data          : pandas dataframe
                        The data for each value of the criterion column and
                        the reference criterion.
        
        criterion     : str
                        The column name to calculate the statistics for.
        
        ref_criterion : str
                        The value of criterion to serve as reference.
                                  
        
        Returns
        ---------------
        ranked_df : pandas dataframe
                    For presentation of the ranked values
    '''
    
    # Calculate the mean, median, standard deviation for each value 
    # of the criterion
    criterion_stats = getStatsFor(data, criterion)
    
    # Keep only the median
    criterion_stats.drop(index=['mean','standard deviation'], level=1, inplace=True)

    # Isolate median value for ref_criterion
    row_ref_criterion = criterion_stats.loc[ref_criterion,:]

    # Calculate for each value of the criterion the difference in %
    # between the median value for the criterion and the reference
    # criterion
    criterion_stats = (criterion_stats.sub(row_ref_criterion).div(row_ref_criterion)*100)

    # Remove the ref_criterion row from dataframe
    criterion_stats.drop(index=[ref_criterion], inplace=True)

    # Create new presentation dataframe
    ranked_df = pd.DataFrame()

    # Calculate the mean difference on all indicators
    ranked_df['Médianes Écarts Tous Indicateurs (%)'] = criterion_stats.median(axis=1)

    # Assemble for presentation
    
    ranked_df = ranked_df.sort_values(['Médianes Écarts Tous Indicateurs (%)'], ascending=False)\
                         .reset_index()\
                         .drop(columns=['grandeur stat'])

    # Add Rank column and set it as index
    ranked_df['RANK']=ranked_df['Médianes Écarts Tous Indicateurs (%)'].rank()
    ranked_df.set_index('RANK', inplace=True)
    
    return ranked_df

#------------------------------------------

def getDifferenceWithCriterionPercent(data, ref_criterion):
    ''' 
        Calculate the difference in % with a reference row.
        
        Parameters
        ----------------
        data          : pandas dataframe
                        The data for each value of the criterion column.
                    
        ref_criterion : str
                        The reference to calculate the difference against.
                                  
        
        Returns
        ---------------
        data_per_criterion : pandas dataframe containing the difference in 
                             % compared with the criterion
    '''
    
    x = pd.DataFrame()
    
    # Group data by indicator
    for ind, data_df in data.groupby(['Indicator Code']):
        # Récupère le row FR
        row_ref_criterion = data_df.loc[data_df['Country Name'] == ref_criterion].loc[:,'2000':]
        
        # Obtenir écarts en % avec les valeurs des autres régions / pays                
        y = pd.concat([data_df.loc[:,'Indicator Code':'Long definition'],
                       data_df.loc[:,'2000':].sub(row_ref_criterion).abs().div(row_ref_criterion)*100],
                       axis=1)
        
        x = pd.concat([x, y])  
        
    # x contient la dataframe des écarts en % par pays par rapport au pays ref_criterion
    
    return x

#------------------------------------------

def getRankByMedianOfTheAbsDifferenceWith(data, criterion, ref_criterion):
    ''' 
        Ranks the given values of a criterion in terms of least 
        absolute mean difference in median values compared to that 
        of a reference criterion.
        
        Parameters
        ----------------
        data          : pandas dataframe
                        The data for each value of the criterion column and
                        the reference criterion.
        
        criterion     : str
                        The column name to calculate the statistics for.
        
        ref_criterion : str
                        The value of criterion to serve as reference.
                                  
        
        Returns
        ---------------
        ranked_df : pandas dataframe
                    For presentation of the ranked values
    '''
    
    # Calculate the difference for each country with the country of reference
    criterion_stats = getDifferenceWithCriterionPercent(data, ref_criterion)

    # Remove the ref_criterion row from dataframe
    criterion_stats = criterion_stats[criterion_stats['Country Name']!=ref_criterion]

    # Create new presentation dataframe
    ranked_df = pd.DataFrame()

    # Calculate the mean difference on all indicators
    for country, data_df in criterion_stats.groupby(['Country Name']):
        tmp_data = data_df.copy()
        tmp_data['Médiane de la médiane des écarts (%)'] = tmp_data.loc[:, '2000':].median(axis=1).median()
                
        tmp_data.loc[:,['Country Name', 'Médiane de la médiane des écarts (%)', 'Region']]
        ranked_df = pd.concat([ranked_df, tmp_data.iloc[0,:]], axis=1)
        
    ranked_df = ranked_df.T.sort_values(by='Médiane de la médiane des écarts (%)').reset_index(drop=True)
        
    return ranked_df

#------------------------------------------

def getDataReadyForMissingValuesPlot(data, criterion):
    criterion_percent_missing_df = pd.DataFrame()

    for column, data_df in data.groupby(criterion):
        
        tmp_df = pd.DataFrame()

        tmp_df['Percent Missing'] = [getMissingValuesPercentPer(data_df, 'Indicator Code')\
                                     ['Percent Missing'].mean()]
        
        tmp_df['Percent Filled'] = 100 - tmp_df['Percent Missing']
        tmp_df['Total'] = 100
        tmp_df[criterion] = column
        
        criterion_percent_missing_df = pd.concat([criterion_percent_missing_df, tmp_df])

    return criterion_percent_missing_df


##########################################
#                                        #
# Fonctions de graphiques                #
#                                        #
##########################################

def plotPercentageMissingValuesFor(data, column, long, larg):
    ''' 
        Plots the proportions of filled / missing values for each unique value
        in column as a stacked horizontal bar chart.
        
        Parameters
        ----------------
        data   : pandas dataframe with: 
                   - a column column
                   - a column "Percent Filled"
                   - a column "Percent Missing"
                   - a column "Total"
                                 
        column : str
                 The column to serve as the y axis for the plot
                                 
        long   : int 
                 The length of the figure for the plot
        
        larg   : int
                 The width of the figure for the plot
                                  
        
        Returns
        ---------------
        -
    '''
    
    data_to_plot = getDataReadyForMissingValuesPlot(data, column)\
                   .sort_values("Percent Missing", ascending=False).reset_index()

    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50


    sns.set(style="whitegrid")

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(long, larg))
    
    plt.title("PROPORTIONS DE VALEURS RENSEIGNÉES / NON-RENSEIGNÉES PAR " + column.upper(),
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)
        
    # Plot the Total values
    sns.set_color_codes("pastel")
    
    b = sns.barplot(x="Total", y=column, data=data_to_plot,label="non renseignées", color="b", alpha=0.3)
    b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
    _, ylabels = plt.yticks()
    _, xlabels = plt.xticks()
    b.set_yticklabels(ylabels, size=TICK_SIZE)


    # Plot the Percent Filled values
    sns.set_color_codes("muted")

    c = sns.barplot(x="Percent Filled", y=column, data=data_to_plot,label="renseignées", color="b")
    c.set_xticklabels(c.get_xticks(), size = TICK_SIZE)
    c.set_yticklabels(ylabels, size=TICK_SIZE)
    

    # Add a legend and informative axis label
    ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, ncol=1, frameon=True,
             fontsize=LEGEND_SIZE)
    
    ax.set(ylabel=column,xlabel="Pourcentage de valeurs (%)")
    
    lx = ax.get_xlabel()
    ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
    
    ly = ax.get_ylabel()
    ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:2d}'.format(int(x)) + '%'))
    ax.tick_params(axis='both', which='major', pad=TICK_PAD)
    
    sns.despine(left=True, bottom=True)

    # Export the figure
    plt.savefig('valeursRenseignees_'+column+'.png')
    
    # Display the figure
    plt.show()

#------------------------------------------

def plotIndicatorFrequencyForTopic(data, long, larg):
    ''' 
        Plots the proportions of indicators per topic in % for each threshold
        as a stacked horizontal bar chart
        
        Parameters
        ----------------
        data   : pandas dataframe with:
                    - an index "Seuil" containing the different threshold 
                      values
                    - a column per topic containing the cumulated % of 
                      indicators (the current topic + the value for the previous
                      column) for that topic for each threshold value
                 
                                 
        long   : int 
                 The length of the figure for the plot
        
        larg   : int
                 The width of the figure for the plot
                                  
        
        Returns
        ---------------
        -
    '''
    
    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 30
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 30
    
    # Reset index to access the Seuil as a column
    data_to_plot = data.reset_index()
    
    sns.set(style="whitegrid")
    palette = sns.husl_palette(len(data.columns))
    
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(long, larg))

    plt.title("RÉPARTITION DES INDICATEURS PAR TOPIC SUIVANT LE SEUIL",
              fontweight="bold",fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Get the list of topics from the columns of data
    column_list = list(data.columns)
    
    # Create a barplot with a distinct color for each topic
    for idx, column in enumerate(reversed(column_list)):        
        color = palette[idx]
        b = sns.barplot(x=column, y="Seuil", data=data_to_plot, label=str(column), orient="h", color=color)
        
        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
        _, ylabels = plt.yticks()
        b.set_yticklabels(ylabels, size=TICK_SIZE)

        
    # Add a legend and informative axis label    
    
    ax.legend(bbox_to_anchor=(0,-1.2,1,0.2), loc="lower left", mode="expand", 
              borderaxespad=0, ncol=1, frameon=True, fontsize=LEGEND_SIZE)

    ax.set(ylabel="Seuil de manquants (%)",xlabel="% d'indicateurs")
    
    lx = ax.get_xlabel()
    ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
    
    ly = ax.get_ylabel()
    ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:2d}'.format(int(x))))

    ax.tick_params(axis='both', which='major', pad=TICK_PAD)
    
    sns.despine(left=True, bottom=True)

    # Export the figure
    plt.savefig('repartitionIndicateursTopicSeuil.png')
    
    # Display the figure
    plt.show()

#------------------------------------------

def plotHistogram(data_to_plot, long, larg):
    ''' 
        Plots a vertical bar chart of the number of indicators per topic 
        for each threshold
        
        Parameters
        ----------------
        data   : pandas dataframe with: 
                   - a column "Seuil" containing the different threshold 
                     values
                   - a column "Total" containing the total number of 
                     indicators for each threshold value
                                 
        long   : int 
                 The length of the figure for the plot
        
        larg   : int
                 The width of the figure for the plot
                                  
        
        Returns
        ---------------
        -
    '''
    
    TITLE_SIZE = 35
    TITLE_PAD = 60
    TICK_SIZE = 30
    TICK_PAD = 30
    LABEL_SIZE = 30
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    
    sns.set(style="whitegrid")

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(long, larg))

    plt.title("NOMBRE D'INDICATEURS PAR SEUIL",fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Create bar chart for each threshold
    sns.set_color_codes("pastel")
    b = sns.barplot(x="Total", y="Seuil", data=data_to_plot,
                label="Nb indicateurs", color="b", orient='h')
    
    b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
    
    _, ylabels = plt.yticks()
    b.set_yticklabels(ylabels, size=TICK_SIZE)
    
    
    # Add a legend and informative axis label
    ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, ncol=1, 
                              frameon=True, fontsize=LEGEND_SIZE)
    
    ax.set(ylabel="Seuil de manquants %", xlabel="Nb Indicateurs")
    
    lx = ax.get_xlabel()
    ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

    ly = ax.get_ylabel()
    ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ax.tick_params(axis='both', which='major', pad=TICK_PAD)
    
    sns.despine(left=True, bottom=True)
    
    # Export the figure
    plt.savefig('nombreIndicateursSeuil.png')
    
    # Display the figure
    plt.show()

#------------------------------------------

def plotRepartitionTopics(data, long, larg):
    ''' 
        Plots a pie chart of the proportion of indicators per topic 
        in data
        
        Parameters
        ----------------
        data   : pandas dataframe with at least:
                     - a column "Topic"
                                  
        long   : int 
                 The length of the figure for the plot
        
        larg   : int
                 The width of the figure for the plot
                 
        Returns
        ---------------
        -
    '''
    
    TITLE_SIZE = 35
    TITLE_PAD = 60

    # Initialize the figure
    f, ax = plt.subplots(figsize=(long, larg))


    # Set figure title
    plt.title("RÉPARTITION DES INDICATEURS PAR TOPIC", fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)
       
    # Put everything in bold
    plt.rcParams["font.weight"] = "bold"


    # Create pie chart for topics
    a = data["Topic"].value_counts(normalize=True).plot(kind='pie', 
                                                        autopct=lambda x:'{:2d}'.format(int(x)) + '%', 
                                                        fontsize =30)
    # Remove y axis label
    ax.set_ylabel('')
    
    # Make pie chart round, not elliptic
    plt.axis('equal') 
    
    # Save the figure 
    plt.savefig('indicateursParTopic.png')
    
    # Display the figure
    plt.show()

#------------------------------------------

def prepareForComparisonPlot(data):
    ''' 
        Divide the data by indicator
        
        Parameters
        ----------------
        data   : pandas dataframe with at least:
                   - a column "Indicator Code"
                                  
        
        Returns
        ---------------
        data_indicator : dictionary with indicators as keys and
                         pandas dataframe as values
                         
    '''
    
    data_indicator = {}

    # Group data by indicator and rearrange inside a dictionary
    for indicator, data_df in data.groupby(['Indicator Code']):  
        if indicator not in data_indicator:
            data_indicator[indicator] = data_df
        else:
            data_indicator[indicator] = pd.concat([data_indicator[indicator], data_df])
    
    return data_indicator

#------------------------------------------

def plotTimeSeriesComparison(data, region, long, larg):
    ''' 
        Displays, for each indicator in data, a timeseries plot with error bands 
        of the difference in % of the values for each region compared to a region 
        of reference.
        
        Parameters
        ----------------
        data   : pandas dataframe
                 The data for each region and region of reference for
                 each indicator
        
        region : str
                 The name of the region of reference
                                 
        long   : int 
                 The length of the figure for the plot
        
        larg   : int
                 The width of the figure for the plot
                                  
        
        Returns
        ---------------
        -
    '''
    
    TITLE_SIZE = 50
    SUBTITLE_SIZE = 30
    LEGEND_SIZE = 30
    
    REGION = region.upper()

    #--------------------------------------------------------------
    # Arange the data in a dictionary with indicators as keys
    # and dataframe as values
    #--------------------------------------------------------------
    
    data_indicator = prepareForComparisonPlot(data)
    
    
    #-------------------------------------------------------------
    # Calculate for each indicator and each (region, country) the 
    # difference in % between the values and the values of the
    # region of reference and arrange the data for plotting.
    #-------------------------------------------------------------
    
    data_final = {}

    for indicator, data_ind_df in data_indicator.items():
        
        #---------------------------------------------------------
        # 0. Get data for the region of reference and rearrange it 
        #    for the future calculations.  
        #---------------------------------------------------------
        
        FR_data = data_ind_df[data_ind_df['Region'] == REGION].copy()
        
        ## Keep only the years columns and arrange to have them as a column
        FR_data = FR_data.loc[:, '2000':'2020'].T.reset_index().rename(columns={"index":"Years"})
        
        ## Add "Region" and "Country Name" columns
        FR_data["Region"] = REGION
        FR_data["Country Name"] = region
        
        # Rename the value column "Value"
        FR_data.columns.values[1] = "Value"
        
        
        
        #----------------------------------------------------------
        # 1. For each (region, country), rearrange the rows and 
        #    columns for the plot function.
        #----------------------------------------------------------
        
        for (region, country), data_df in data_ind_df.groupby(['Region', 'Country Name']):
            
            # Get data for each (region, country)
            tmp_df = data_df.copy()
            
            # Keep only the years columns and arrange to have them as a column
            tmp_df = tmp_df.loc[:, '2000':'2020'].T.reset_index().rename(columns={"index":"Years"})
            
            # Add "Region" and "Country Name" columns
            tmp_df["Region"] = region
            tmp_df["Country Name"] = country
            
            # Rename the value column "Value"
            tmp_df.columns.values[1] = "Value"
            
            #-------------------------------------------------------
            # 2. Calculate the difference in % between the values 
            #    for the region and the region of reference
            #-------------------------------------------------------
            
            tmp_df["Value"] = tmp_df["Value"].sub(FR_data["Value"]).abs().div(FR_data["Value"])*100
            
            #-------------------------------------------------------
            # 3. Assemble a dataframe for all the (region, country) 
            #    for a given indicator and store it in a dictionary 
            #    with the indicator code as key
            #-------------------------------------------------------
            
            if indicator not in data_final:
                data_final[indicator] = tmp_df
            else:
                data_final[indicator] = pd.concat([data_final[indicator], tmp_df])
                
                
                
    #--------------------------------------------------------------
    # Create for each indicator a timeseries plot with the values 
    # for all the regions and the region of reference.
    #--------------------------------------------------------------

    sns.set(style="whitegrid")
    
    f, axes = plt.subplots(figsize=(long, larg), ncols=3, nrows=3)    
    
    f.suptitle("ÉCART ZONES GÉOGRAPHIQUES - FRANCE",fontweight="bold",
               fontsize=TITLE_SIZE)

    # For each indicator
    for (indicator, data_to_plot), ax in zip(data_final.items(), axes.flat):
                    
        # Set title of the subplot
        ax.set_title("Indicator " + indicator, fontsize=SUBTITLE_SIZE, fontweight="bold")

        # Create lineplot with all  regions
        color_palette = {'East Asia & Pacific':'coral',
                         'Europe & Central Asia': 'mediumpurple',
                         'North America': 'mediumseagreen',
                         'Latin America & Caribbean':'turquoise',
                         'Middle East & North Africa': 'gold',
                         'Sub-Saharan Africa': 'darkgoldenrod',
                         'South Asia': 'r',
                         'FRANCE':'fuchsia'}
        
        sns.lineplot(x="Years", y="Value", hue="Region", 
                     data=data_to_plot, ax=ax, 
                     palette=color_palette)
        
        # Add a legend and informative axis label
        ax.get_legend().remove()

        ax.set(ylabel="Écart (%)",
               xlabel="Années ")
        sns.despine(left=True, bottom=True)
    
        handles, labels = ax.get_legend_handles_labels()

        f.legend(handles, labels, loc="lower center", ncol=3, frameon=True, prop={'size': LEGEND_SIZE})
        
    # Display the plots
    plt.show()

#------------------------------------------


def plotLineplotComparisonCriteria(data, criterion, ref_criterion, long, larg):
    ''' 
        Displays, for each indicator in data, a line plot of the absolute mean 
        difference in % of the values for each unique criterion compared to a 
        reference value for the criterion.
        
        Parameters
        ----------------
        data          : pandas dataframe
                        The data for each region and region of reference for
                        each indicator
        
        criterion     : str
                        The name of the column by which to group by
                    
        ref_criterion : str
                        The value of the reference criterion
                                 
        long          : int 
                        The length of the figure for the plot
        
        larg          : int
                        The width of the figure for the plot
                                  
        
        Returns
        ---------------
        -
    '''
    
    
    #--------------------------------------------------------------
    # Arrange the data in a dictionary with indicators as keys
    # and dataframe as values
    #--------------------------------------------------------------
    
    data_indicator = prepareForComparisonPlot(data)  
    
    # Rearrange the rows and columns for the plotting function
    data_lineplot_indicator = {}

    
    for indicator, data_df in data_indicator.items():
        
        #----------------------------------------------------------
        # Calculate the mean difference in % between the values for 
        # the region and the region of reference, and store the new 
        # dataframe in a dictionary with the indicator as key.
        #----------------------------------------------------------
        
        tmp_df = data_df.groupby([criterion]).agg("median").T
        
        col_criterion = tmp_df[ref_criterion]
        
        
        data_lineplot_indicator[indicator] = tmp_df.sub(col_criterion, axis=0)\
                                                   .abs().div(col_criterion, axis=0)*100
     
    
    #--------------------------------------------------------------
    # Create for each indicator a line plot with the values 
    # for all the regions and the region of reference.
    #--------------------------------------------------------------    
    
    sns.set(style="whitegrid")
    
    nb_of_rows = int(len(data_indicator)/3) if int(len(data_indicator)/3)>0 else 1
    nb_of_cols = int(3 if nb_of_rows > 1 else 1)
    
    f, axes = plt.subplots(figsize=(long, larg), ncols=nb_of_cols, nrows=nb_of_rows)
    
    if nb_of_rows == 1:
        # For each indicator
        for indicator, data_to_plot in data_lineplot_indicator.items():
            
            TITLE_SIZE = 20
            TITLE_PAD = 10
            LEGEND_SIZE = 15

            TICK_SIZE = 12
            TICK_PAD = 10
            LABEL_SIZE = 20
            LABEL_PAD = 10
    
            # Set title
            plt.title("INDICATEUR " + indicator, fontweight="bold", 
                      fontsize=TITLE_SIZE, pad=TITLE_PAD)

            # Create lineplot for each region
            color_palette = {'Austria':'navy',
                         'United Kingdom': 'dodgerblue',
                         'Canada': 'mediumseagreen',
                         'Belgium':'springgreen',
                         'Hungary': 'gold',
                         'Germany': 'mediumpurple',
                         'Spain': 'crimson',
                         'Finland': "plum",
                         'Slovenia':'salmon',
                         'Ireland':'brown',
                         'France':'fuchsia'}
            
            b = sns.lineplot(data=data_to_plot, palette=color_palette, linewidth=2.5, dashes=False)
            
            b.set_yticklabels(b.get_yticks(), size = TICK_SIZE)

            
            # Add legend and informative axis label
            axes.legend(bbox_to_anchor=(0,-0.5,1,0.2), loc="lower center", borderaxespad=0, ncol=3, 
                              frameon=True, fontsize=LEGEND_SIZE)
            
            axes.set(ylabel="Écart (%)", xlabel="Années")
            
            lx = axes.get_xlabel()
            axes.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

            ly = axes.get_ylabel()
            axes.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
            axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '{:d}'.format(int(y))))

            axes.tick_params(axis='both', which='major', pad=TICK_PAD)
            
            sns.despine(left=True, bottom=True)   
            
    else:
        
        TITLE_SIZE = 50
        TITLE_PAD = 40
        SUBTITLE_SIZE = 30
        LEGEND_SIZE = 30

        TICK_SIZE = 30
        TICK_PAD = 30
        LABEL_SIZE = 30
        LABEL_PAD = 30
    
        f.suptitle("ÉCART ZONES GÉOGRAPHIQUES - FRANCE",fontweight="bold",fontsize=TITLE_SIZE)

        # For each indicator
        for (indicator, data_to_plot), ax in zip(data_lineplot_indicator.items(), axes.flat):

            # Set title
            ax.set_title("Indicator " + indicator, fontsize=SUBTITLE_SIZE, fontweight="bold")

            # Create lineplot for each region
            color_palette = {'East Asia & Pacific':'coral',
                             'Europe & Central Asia': 'mediumpurple',
                             'North America': 'mediumseagreen',
                             'Latin America & Caribbean':'turquoise',
                             'Middle East & North Africa': 'gold',
                             'Sub-Saharan Africa': 'darkgoldenrod',
                             'South Asia': 'r',
                             'FRANCE':'fuchsia'}
            
            sns.lineplot(data=data_to_plot, palette=color_palette, linewidth=2.5, dashes=False, ax=ax)

            # Add legend and informative axis label
            ax.get_legend().remove()

            ax.set(ylabel="Écart (%)",
                   xlabel="Années")
            sns.despine(left=True, bottom=True)
            
            handles, labels = ax.get_legend_handles_labels()

            f.legend(handles, labels, loc="lower center", ncol=3, frameon=True, prop={'size': LEGEND_SIZE})

    plt.show()

#------------------------------------------

def plotLineplotComparisonCriteriaNoAbs(data, criterion, ref_criterion, long, larg):
    ''' 
        Displays, for each indicator in data, a line plot of the mean difference 
        in % of the values for each region compared to a region of reference.
        
        Input
        ----------------
        data          : pandas dataframe
                        The data for each region and region of reference for
                        each indicator
        
        criterion     : str
                        The name of the column by which to group by
                    
        ref_criterion : str
                        The value of the reference criterion
                                 
        long          : int 
                        The length of the figure for the plot
        
        larg          : int
                        The width of the figure for the plot
                                  
        
        Output
        ---------------
        -
    '''
    
    
    #--------------------------------------------------------------
    # Arrange the data in a dictionary with indicators as keys
    # and dataframe as values
    #--------------------------------------------------------------
    
    data_indicator = prepareForComparisonPlot(data)  
    
    # Rearrange the rows and columns for the plotting function
    data_lineplot_indicator = {}

    
    for indicator, data_df in data_indicator.items():
        
        #----------------------------------------------------------
        # Calculate the mean difference in % between the values for 
        # the region and the region of reference, and store the new 
        # dataframe in a dictionary with the indicator as key.
        #----------------------------------------------------------
        
        tmp_df = data_df.groupby([criterion]).agg("mean").T
        
        col_criterion = tmp_df[ref_criterion]
        
        
        data_lineplot_indicator[indicator] = tmp_df.sub(col_criterion, axis=0)\
                                                   .div(col_criterion, axis=0)*100
     
    
    #--------------------------------------------------------------
    # Create for each indicator a line plot with the values 
    # for all the regions and the region of reference.
    #--------------------------------------------------------------    
    
    sns.set(style="whitegrid")
    
    nb_of_rows = int(len(data_indicator)/3) if int(len(data_indicator)/3)>0 else 1
    nb_of_cols = int(3 if nb_of_rows > 1 else 1)
    
    f, axes = plt.subplots(figsize=(long, larg), ncols=nb_of_cols, nrows=nb_of_rows)
    
    if nb_of_rows == 1:
        # For each indicator
        for indicator, data_to_plot in data_lineplot_indicator.items():

            TITLE_SIZE = 20
            TITLE_PAD = 10
            LEGEND_SIZE = 15

            TICK_SIZE = 12
            TICK_PAD = 10
            LABEL_SIZE = 20
            LABEL_PAD = 10
    
            # Set title
            plt.title("INDICATEUR " + indicator, fontweight="bold", 
                      fontsize=TITLE_SIZE, pad=TITLE_PAD)

            # Create lineplot for each region
            color_palette = {'Austria':'navy',
                         'United Kingdom': 'dodgerblue',
                         'Canada': 'mediumseagreen',
                         'Belgium':'springgreen',
                         'Hungary': 'gold',
                         'Germany': 'mediumpurple',
                         'Spain': 'crimson',
                         'Finland': "plum",
                         'Slovenia':'salmon',
                         'Ireland':'brown',
                         'France':'fuchsia'}
            
            b = sns.lineplot(data=data_to_plot, palette=color_palette, linewidth=2.5, dashes=False)
            
            b.set_yticklabels(b.get_yticks(), size = TICK_SIZE)

            
            # Add legend and informative axis label
            axes.legend(bbox_to_anchor=(0,-0.25,1,0.2), loc="lower center", borderaxespad=0, ncol=3, 
                              frameon=True, fontsize=LEGEND_SIZE)
            #axes.legend(ncol=2, loc="upper right", frameon=True, fontsize=LEGEND_SIZE)
            
            axes.set(ylabel="Écart (%)", xlabel="Années")
            
            lx = axes.get_xlabel()
            axes.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

            ly = axes.get_ylabel()
            axes.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
            axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '{:d}'.format(int(y))))

            axes.tick_params(axis='both', which='major', pad=TICK_PAD)
            
            sns.despine(left=True, bottom=True)   
            
    else:
        
        TITLE_SIZE = 40
        TITLE_PAD = 40
        SUBTITLE_SIZE = 30
        LEGEND_SIZE = 30

        TICK_SIZE = 30
        TICK_PAD = 30
        LABEL_SIZE = 30
        LABEL_PAD = 30
    
        f.suptitle("ÉCART ZONES GÉOGRAPHIQUES - FRANCE",fontweight="bold",fontsize=TITLE_SIZE)

        # For each indicator
        for (indicator, data_to_plot), ax in zip(data_lineplot_indicator.items(), axes.flat):

            # Set title
            ax.set_title("Indicator " + indicator, fontsize=SUBTITLE_SIZE, fontweight="bold")

            # Create lineplot for each region
            color_palette = {'East Asia & Pacific':'coral',
                             'Europe & Central Asia': 'mediumpurple',
                             'North America': 'mediumseagreen',
                             'Latin America & Caribbean':'turquoise',
                             'Middle East & North Africa': 'gold',
                             'Sub-Saharan Africa': 'darkgoldenrod',
                             'South Asia': 'r',
                             'FRANCE':'fuchsia'}
            
            sns.lineplot(data=data_to_plot, palette=color_palette, linewidth=2.5, dashes=False, ax=ax)

            # Add legend and informative axis label
            ax.get_legend().remove()

            ax.set(ylabel="Écart (%)",
                   xlabel="Années")
            sns.despine(left=True, bottom=True)
            
            handles, labels = ax.get_legend_handles_labels()

            f.legend(handles, labels, loc="lower center", ncol=3, frameon=True, prop={'size': LEGEND_SIZE})

    plt.show()

#------------------------------------------

def plotLmplot(data, long, larg):
    ''' 
        Displays, for the indicator in data, a lmplot of the value
        for each country.
        
        Parameters
        ----------------
        data          : pandas dataframe
                        The data for each region and region of reference for
                        the indicator
                                 
        long          : int 
                        The length of the figure for the plot
        
        larg          : int
                        The width of the figure for the plot
                                  
        
        Returns
        ---------------
        -
    '''
    
     
    TITLE_SIZE = 30
    TITLE_PAD = 60
    TICK_SIZE = 20
    TICK_PAD = 30
    LABEL_SIZE = 30
    LABEL_PAD = 30
    LEGEND_SIZE = 15
    
    sns.set(style="whitegrid")

    # Create bar chart for each threshold
    color_palette = {'Austria':'navy',
                     'United Kingdom': 'dodgerblue',
                     'Canada': 'mediumseagreen',
                     'Belgium':'springgreen',
                     'Hungary': 'gold',
                     'Germany': 'mediumpurple',
                     'Spain': 'crimson',
                     'Finland': "plum",
                     'Slovenia':'salmon',
                     'Ireland':'brown',
                     'France':'fuchsia'}
    
    b = sns.lmplot(x="Years", y="Value", hue="Country", data=data, height=10, legend=False, palette=color_palette)
    
    ax = plt.gca()
    ax.set_title("Indicator SP.POP.GROW", fontweight="bold", fontsize=TITLE_SIZE, pad=TITLE_PAD)
    sns.despine(left=True, bottom=True)
    
    ax.set(ylabel="Taux de croissance (%)", xlabel="Années")
    lx = ax.get_xlabel()
    ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ly = ax.get_ylabel()
    ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
    
    ax.tick_params(axis='both', which='major', pad=TICK_PAD)

    ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, ncol=1, 
                              frameon=True, fontsize=LEGEND_SIZE)
    
    _, xlabels = plt.xticks()
    b.set_xticklabels(xlabels, size=TICK_SIZE)
    
    _, ylabels = plt.yticks()
    b.set_yticklabels(ylabels, size=TICK_SIZE)
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

    sns.despine(left=True, bottom=True)   

#------------------------------------------

def plotLineplotIndicators(data, indicators, long, larg):
    ''' 
        Displays, for the indicators in indicators, a lineplot of 
        the values for each country.
        
        Parameters
        ----------------
        data          : pandas dataframe
                        The data for each region and region of reference for
                        the indicator
                                 
        long          : int 
                        The length of the figure for the plot
        
        larg          : int
                        The width of the figure for the plot
                                  
        
        Returns
        ---------------
        -
    '''
    
    data_to_plot = pd.DataFrame()
    
    projected_data_country_ind = pd.DataFrame()

    for country, data_country in data.groupby(['Country Name']):

        for indicator in indicators:
            
            projected_data_country_ind = data_country[data_country['Indicator Code']==indicator]\
                                         .loc[:,'2020':'2100'].reset_index()

            projected_data_country_ind = projected_data_country_ind.T.reset_index().iloc[1:,:]\
                                         .rename(columns={'index':'Years', 0:'Value'}) 
            
            projected_data_country_ind['Country']= country
            projected_data_country_ind['Indicator Code'] = indicator

            data_to_plot = pd.concat([data_to_plot, projected_data_country_ind])

    data_to_plot['Years']=data_to_plot['Years'].astype(float)
    data_to_plot['Value']=data_to_plot['Value'].astype(float)
    
    # Plot the responses for different events and regions
    TITLE_SIZE = 35
    TITLE_PAD = 60
    TICK_SIZE = 30
    TICK_PAD = 30
    LABEL_SIZE = 30
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5
    
    f, ax = plt.subplots(figsize=(long, larg))

    plt.title("PROJECTIONS : ÉVOLUTION NIVEAU D'ÉTUDE POPULATION ACTIVE",fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)
    
    if len(indicators)==1:
        color_palette = {'Austria':'navy',
                         'United Kingdom': 'dodgerblue',
                         'Canada': 'mediumseagreen',
                         'Belgium':'springgreen',
                         'Hungary': 'gold',
                         'Germany': 'mediumpurple',
                         'Spain': 'crimson',
                         'Finland': "plum",
                         'Slovenia':'salmon',
                         'Ireland':'brown',
                         'France':'fuchsia'}
        
        b = sns.lineplot(x="Years", y="Value",
                         hue="Country",data=data_to_plot, 
                         palette=color_palette, linewidth=LINE_WIDTH)
    else:
        color_palette = {'PRJ.ATT.15UP.1.MF':'turquoise',
                         'PRJ.ATT.15UP.3.MF': 'orange',
                         'PRJ.ATT.15UP.4.MF':'limegreen'}
        
        b = sns.lineplot(x="Years", y="Value", 
                         hue="Indicator Code", style='Country',
                         data=data_to_plot, palette=color_palette, linewidth=LINE_WIDTH)
    
    b.set_yticklabels(b.get_yticks(), size = TICK_SIZE)
    b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
    
    ax.set(ylabel="% pop active atteignant le niveau d'étude cible", xlabel="Années")
    
    lx = ax.get_xlabel()
    ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))
    
    ly = ax.get_ylabel()
    ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '{:d}'.format(int(y)) + '%'))

    ax.tick_params(axis='both', which='major', pad=TICK_PAD)
            
    ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, ncol=1, 
                              frameon=True, fontsize=LEGEND_SIZE)
    
    sns.despine(left=True, bottom=True)   
    
    plt.show()

#------------------------------------------

def plotBoxplot(data, criterion, long, larg):
    ''' 
        Displays a boxplot of the values for the criterion.
        
        Parameters
        ----------------
        data          : pandas dataframe
                        The data for each region and region of reference for
                        the indicator
                                 
        long          : int 
                        The length of the figure for the plot
        
        larg          : int
                        The width of the figure for the plot
                                  
        
        Returns
        ---------------
        -
    '''
    b = data.copy()
    b[criterion] = 'GLOBAL'

    data_to_plot = pd.concat([data,b])
    
    TITLE_SIZE = 35
    TITLE_PAD = 60
    TICK_SIZE = 25
    TICK_PAD = 30
    LABEL_SIZE = 30
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5
    
    sns.set(style="ticks")

    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(long, larg))
    
    plt.title("DISTRIBUTION DES ÉCARTS RÉGIONAUX",fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)
    
    #ax.set_xscale("log")

    # Plot the orbital period with horizontal boxes
    b = sns.boxplot(x="Médianes Écarts Tous Indicateurs (%)", y=criterion, data=data_to_plot,
                whis=[0, 100], color="lightskyblue")

    _, ylabels = plt.yticks()
    b.set_yticklabels(ylabels, size=TICK_SIZE)
    b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
    
    ax.set(ylabel="Pop Cible (% de la pop active)", xlabel="Années")
    
    lx = ax.get_xlabel()
    ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x)) + '%'))
    
    ly = ax.get_ylabel()
    ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ax.tick_params(axis='both', which='major', pad=TICK_PAD)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)

