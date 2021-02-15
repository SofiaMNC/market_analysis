# **Going Global: Market Analysis For An EdTech Company**
*Sofia Chevrolat (May 2020)*
___
> **NB** : This project is the first of a series comprising the [syllabus offered by OpenClassrooms in partnership with Centrale Supélec and sanctioned by the Data Scientist diploma - Master level](https://openclassrooms.com/fr/paths/164-data-scientist).


This notebook aims to study the feasability of exploiting the contents of the Data World Bank's _“EdStats All Indicator Query”_ database, consisting of about 4000 indicators on the theme of the education of populations worldwide, as part of a project of international expansion lead by the fictional start-up EdTech Academy. 
Based in France, the company offers online classes and cursus aimed at highschool and university level customers.

Academy would like to explore the following questions: 
- Which countries display a strong customer potential for our services? 
- For each of these countries, how will this customer potential evolve?
- Which of these countries should be given priority to begin with?

Given the questions, this pre-analysis will focus on three points:
- **Access to a computer and internet**: the classes being given online, this point is crucial.
- **Populations studying at highschool or college levels**: the number of people having achieved the required levels of education, without regards for age. Adults past college age could for instance be studying college level classes (career change, new skill acquisition...).
- **The size of the potential market** : the total number of working age people, which is to say the age group 15 years and older.

In this study, the French market will be taken as reference.
Indeed, the initial assumption here is that the company Academy is in a capacity to think about expanding because its business model is successful in its native country. 
Geographical zones and countries whose profiles are similar to that of France shoud therefore be considered more relevant.
___

This notebook is organised as follows:

**0. Setting up**
- 0.1 Loading the necessary libraries
- 0.2 Defining custom functions
- 0.3 Loading the data set

**1. Description and data clean up**
- 1.1 Description of the data set
- 1.2 Assembly and clean up of the data
- 1.3 Quality of the data set
    * 1.3.1 Descriptive data
    * 1.3.2 Projective data

**2. Exploratory Data Pre-Analyse**
- 2.1 Determining the relevant categories of indicators
- 2.2 Determining the relevant indicators
    * 2.2.1 Determining the optimal threshold for missing data (descriptive dataframe)
    * 2.2.2 Selecting the indicators of interest
- 2.3 Determining the geographical zones of interest
    * 2.3.1 Setting up
    * 2.3.2 Comparing the geographical zones' profiles
    * 2.3.3 Selecting the geographical zones of interest
- 2.4 Determining the most interesting countries
    * 2.4.1 Statistics for all countries
    * 2.4.2 Selecting the countries of interest

**3. Conclusion : validating the data set for the problematic at hand, recommandations and points of improvement**

_____________

## Requirements
This assumes that you already have an environment allowing you to run Jupyter notebooks.
The libraries used otherwise are listed in requirements.txt

_____________

## Usage

1. Download the dataset from [OpenClassrooms](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Données+éducatives/Projet+Python_Dataset_Edstats_csv.zip), and place the files under Sources/.

2. If you would like to know more about the data set itself, the sources are also accessible directly on the [World Bank website](https://datacatalog.worldbank.org/dataset/education-statistics)

3. Run the following in your terminal to install all required libraries : 
```bash
pip3 install -r requirements.txt
```
4. Run the notebook.

_______

## Results

For a complete presentation of the results of this pre-analysis, please see the powerpoint presentation. 

> NOTE: The presentation is in **French**. 
