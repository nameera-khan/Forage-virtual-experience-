# Forage-virtual-experience-
Virtual experience projects on forage.com

### British Airways 
1. Text analysis - Text analysis for insights using the customer reviews on https://www.airlinequality.com/airline-reviews/british-airways. This project was performed as a part of the virtual experience programme for British airways. It involves web scrapping, data manipulation and generating insights using word clouds to analyse the top words used in reviews. The purpose behind such an analysis would assist in focussing on areas that is satisfactory according to the customers or areas that need improvement. Using modules such as scikit-learn, matplotlib and BeautifulSoup.
2. Predictive analysis - The customer data given by British Airways is available at BA's job simulation at forage.com. The dataset consists of customer booking data with a high imbalance ratio. This notebook consists of the usage of RandomForest to classify booking prediction in binary classes 0 and 1. By utilising the Random oversampling method the f1-score obtained for the test dataset is 0.15. By iterating over different oversampling strategies, the f1-score differed from the original f1 score by approx. 100%. It is important to note that while building the perfect model is not under the scope of this simulation, by obtaining variables that withdraw the necessary knowledg needed to predict the minority class is imperative.
Additionally, this model proves the chi-squared test by showcasing the importance given to the 'purchase_lead' variable in association to customer bookings. However, the dataset lacks the dexterity of a customer data with the absense of variables such as ticket price and so on, which can be considered as influencing factors to a customer's decision as mentioned above. 

### Quantium 
Quantium-data-analysis
* Exploratory data analysis of market dataset
The analysis consists of fields with customer transaction details and customer behaviour. The first task involves exploratory data analysis of the dataset which includes the following steps:
* Importing necessary modules
* Summary statistics
* Data visualisation for customer insights
* Deep diving into trends set by the transactions on monthly basis.
* Task 2 - Experimentation and uplift test This task involves choosing a null-hypothesis against 3 trial stores that have been selected by the client.
* The control stores have been chosen based on Pearson correlation constant calculated  and magnitude difference for total sales and number of customers in each store. The store with a final score that is close to the trial store was chosen as the control store.
* The three trial stores chosen by the client were : 77,86 and 88.
* The assessments for these stores were done against the control stores to generate the sales during a trial period of 2019-02 and 2019-04.
Libraries used: Pandas, Matplotlib, Seaborn

### Cognizant 
* AI for Cognizant’s Data Science team. 
* Conducted exploratory data analysis using Python for one of Cognizant’s technology-led clients, Gala Groceries.
* Task 1- Exploratory Data analysis of the dataset presented by Gala Groceries using Plotly's interactive library for temporal trends in sales.
* Task 2- Built a model using the additional IoT sensor temperature data and predict estimated stock levels for procurement. Random Forest Regressor with Mean Absolute Error as evaluation metric.
* Task 3 - Prepare a Python module that contains code to train a model and output the performance metrics for the Machine Learning engineering team.
* Feature importance visualisation where temperature was one of the influencial variables. 
Libraries used: Pandas,Numpy, Maplotlib, Plotly, Scikit-Learn
