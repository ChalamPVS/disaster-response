# Disaster Response Pipeline Project

The project analyzes disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data set containes real messages that were sent during disaster events. This dataset is used to build an ML model using NLP techniques to automatically categorize the disaster events so that the messages can then be sent to an appropriate disaster relief agency. 

There are about 36 different disaster events, wherein a singe message can be categorized to one or more of these events; hence, this is a multiclassification problem. The various categories are:

1. 'related',
2. 'request',
3. 'offer',
4. 'aid_related',
5. 'medical_help',
6. 'medical_products',
7. 'search_and_rescue',
8. 'security',
9. 'military',
10. 'child_alone',
11. 'water',
12. 'food',
13. 'shelter',
14. 'clothing',
15. 'money',
16. 'missing_people',
17. 'refugees',
18. 'death',
19. 'other_aid',
20. 'infrastructure_related',
21. 'transport',
22. 'buildings',
23. 'electricity',
24. 'tools',
25. 'hospitals',
26. 'shops',
27. 'aid_centers',
28. 'other_infrastructure',
29. 'weather_related',
30. 'floods',
31. 'storm',
32. 'fire',
33. 'earthquake',
34. 'cold',
35. 'other_weather',
36. 'direct_report'

## Requirements
- Python 3.5 or higher
- Numpy
- Matplotlib
- Pandas
- NLTK (specifically, 'stopwords', 'punkt', 'wordnet')
- plotly
- Scikit-learn


## Summary
The projects implements three main components:
- ETL Pipeline: 
	- Loads the messages and categories datasets
	- Merges the two datasets
	- Cleans the data
	- Stores it in a SQLite database

- Machine Learning Pipeline using Natural Language Processing
	- Loads data from the SQLite database
	- Splits the dataset into training and test sets
	- Builds a text processing and machine learning pipeline
	- Trains and tunes a model using GridSearchCV
	- Outputs results on the test set
	- Exports the final model as a pickle file

- Deployment of Flask App
	- Deploys a visualization tool 

## Repository Break-down:

- app
	- run.py: a python script using flask for API deployment
	- templates: necessary templates for API deployment
- data
	- disaster_messages.csv: input data containing text messages
	- disaster_categories.csv: input data containing disaster categorization
	- process_data.py: a python script to process the input data
	- DisasterResponse.db: a sql database to store the processed data
- models
	- train_classifier.py: a python script to train an ML model

###Note
`classifier.pkl` is not included due to size constraints of github; instead it is provided in a [google drive](https://drive.google.com/file/d/1PtNLRG9pSgyVX2699QAtlrO5YFFX70yo/view?usp=sharing) 

## Instructions:


1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Results

************************************************************
Category Name: related
Accuracy: 0.81
Classification Report
             precision    recall  f1-score   support

          0       0.67      0.44      0.53      1234
          1       0.84      0.93      0.88      3969
          2       0.27      0.37      0.31        41

avg / total       0.80      0.81      0.80      5244

************************************************************
************************************************************
Category Name: request
Accuracy: 0.89
Classification Report
             precision    recall  f1-score   support

          0       0.90      0.98      0.94      4359
          1       0.83      0.44      0.57       885

avg / total       0.89      0.89      0.88      5244

************************************************************
************************************************************
Category Name: offer
Accuracy: 1.00
Classification Report
             precision    recall  f1-score   support

          0       1.00      1.00      1.00      5219
          1       0.00      0.00      0.00        25

avg / total       0.99      1.00      0.99      5244

************************************************************
************************************************************
Category Name: aid_related
Accuracy: 0.76
Classification Report
             precision    recall  f1-score   support

          0       0.78      0.85      0.81      3134
          1       0.74      0.64      0.69      2110

avg / total       0.76      0.76      0.76      5244

************************************************************
************************************************************
Category Name: medical_help
Accuracy: 0.93
Classification Report
             precision    recall  f1-score   support

          0       0.93      0.99      0.96      4837
          1       0.63      0.10      0.18       407

avg / total       0.91      0.93      0.90      5244

************************************************************
************************************************************
Category Name: medical_products
Accuracy: 0.96
Classification Report
             precision    recall  f1-score   support

          0       0.96      1.00      0.98      5000
          1       0.69      0.10      0.18       244

avg / total       0.95      0.96      0.94      5244

************************************************************
************************************************************
Category Name: search_and_rescue
Accuracy: 0.97
Classification Report
             precision    recall  f1-score   support

          0       0.97      1.00      0.99      5087
          1       0.64      0.06      0.11       157

avg / total       0.96      0.97      0.96      5244

************************************************************
************************************************************
Category Name: security
Accuracy: 0.98
Classification Report
             precision    recall  f1-score   support

          0       0.98      1.00      0.99      5161
          1       0.00      0.00      0.00        83

avg / total       0.97      0.98      0.98      5244

************************************************************
************************************************************
Category Name: military
Accuracy: 0.97
Classification Report
             precision    recall  f1-score   support

          0       0.97      1.00      0.98      5066
          1       0.71      0.10      0.17       178

avg / total       0.96      0.97      0.96      5244

************************************************************
************************************************************
Category Name: child_alone
Accuracy: 1.00
Classification Report
             precision    recall  f1-score   support

          0       1.00      1.00      1.00      5244

avg / total       1.00      1.00      1.00      5244

************************************************************
************************************************************
Category Name: water
Accuracy: 0.96
Classification Report
             precision    recall  f1-score   support

          0       0.96      1.00      0.98      4914
          1       0.88      0.36      0.52       330

avg / total       0.95      0.96      0.95      5244

************************************************************
************************************************************
Category Name: food
Accuracy: 0.94
Classification Report
             precision    recall  f1-score   support

          0       0.95      0.99      0.97      4687
          1       0.83      0.55      0.66       557

avg / total       0.94      0.94      0.94      5244

************************************************************
************************************************************
Category Name: shelter
Accuracy: 0.93
Classification Report
             precision    recall  f1-score   support

          0       0.94      0.99      0.96      4779
          1       0.84      0.31      0.46       465

avg / total       0.93      0.93      0.92      5244

************************************************************
************************************************************
Category Name: clothing
Accuracy: 0.99
Classification Report
             precision    recall  f1-score   support

          0       0.99      1.00      0.99      5170
          1       0.75      0.08      0.15        74

avg / total       0.98      0.99      0.98      5244

************************************************************
************************************************************
Category Name: money
Accuracy: 0.98
Classification Report
             precision    recall  f1-score   support

          0       0.98      1.00      0.99      5136
          1       1.00      0.04      0.07       108

avg / total       0.98      0.98      0.97      5244

************************************************************
************************************************************
Category Name: missing_people
Accuracy: 0.99
Classification Report
             precision    recall  f1-score   support

          0       0.99      1.00      1.00      5200
          1       1.00      0.02      0.04        44

avg / total       0.99      0.99      0.99      5244

************************************************************
************************************************************
Category Name: refugees
Accuracy: 0.97
Classification Report
             precision    recall  f1-score   support

          0       0.97      1.00      0.99      5092
          1       0.57      0.09      0.15       152

avg / total       0.96      0.97      0.96      5244

************************************************************
************************************************************
Category Name: death
Accuracy: 0.96
Classification Report
             precision    recall  f1-score   support

          0       0.96      1.00      0.98      5015
          1       0.78      0.12      0.21       229

avg / total       0.95      0.96      0.95      5244

************************************************************
************************************************************
Category Name: other_aid
Accuracy: 0.87
Classification Report
             precision    recall  f1-score   support

          0       0.87      0.99      0.93      4561
          1       0.52      0.04      0.08       683

avg / total       0.83      0.87      0.82      5244

************************************************************
************************************************************
Category Name: infrastructure_related
Accuracy: 0.93
Classification Report
             precision    recall  f1-score   support

          0       0.93      1.00      0.97      4902
          1       0.00      0.00      0.00       342

avg / total       0.87      0.93      0.90      5244

************************************************************
************************************************************
Category Name: transport
Accuracy: 0.96
Classification Report
             precision    recall  f1-score   support

          0       0.96      1.00      0.98      5012
          1       0.75      0.08      0.14       232

avg / total       0.95      0.96      0.94      5244

************************************************************
************************************************************
Category Name: buildings
Accuracy: 0.95
Classification Report
             precision    recall  f1-score   support

          0       0.95      1.00      0.97      4968
          1       0.77      0.11      0.19       276

avg / total       0.94      0.95      0.93      5244

************************************************************
************************************************************
Category Name: electricity
Accuracy: 0.98
Classification Report
             precision    recall  f1-score   support

          0       0.98      1.00      0.99      5132
          1       0.78      0.06      0.12       112

avg / total       0.98      0.98      0.97      5244

************************************************************
************************************************************
Category Name: tools
Accuracy: 0.99
Classification Report
             precision    recall  f1-score   support

          0       0.99      1.00      1.00      5215
          1       0.00      0.00      0.00        29

avg / total       0.99      0.99      0.99      5244

************************************************************
************************************************************
Category Name: hospitals
Accuracy: 0.99
Classification Report
             precision    recall  f1-score   support

          0       0.99      1.00      0.99      5184
          1       0.00      0.00      0.00        60

avg / total       0.98      0.99      0.98      5244

************************************************************
************************************************************
Category Name: shops
Accuracy: 1.00
Classification Report
             precision    recall  f1-score   support

          0       1.00      1.00      1.00      5227
          1       0.00      0.00      0.00        17

avg / total       0.99      1.00      1.00      5244

************************************************************
************************************************************
Category Name: aid_centers
Accuracy: 0.99
Classification Report
             precision    recall  f1-score   support

          0       0.99      1.00      0.99      5178
          1       0.00      0.00      0.00        66

avg / total       0.97      0.99      0.98      5244

************************************************************
************************************************************
Category Name: other_infrastructure
Accuracy: 0.96
Classification Report
             precision    recall  f1-score   support

          0       0.96      1.00      0.98      5010
          1       0.00      0.00      0.00       234

avg / total       0.91      0.96      0.93      5244

************************************************************
************************************************************
Category Name: weather_related
Accuracy: 0.87
Classification Report
             precision    recall  f1-score   support

          0       0.87      0.95      0.91      3768
          1       0.85      0.64      0.73      1476

avg / total       0.86      0.87      0.86      5244

************************************************************
************************************************************
Category Name: floods
Accuracy: 0.95
Classification Report
             precision    recall  f1-score   support

          0       0.95      1.00      0.97      4806
          1       0.92      0.42      0.57       438

avg / total       0.95      0.95      0.94      5244

************************************************************
************************************************************
Category Name: storm
Accuracy: 0.93
Classification Report
             precision    recall  f1-score   support

          0       0.94      0.99      0.96      4734
          1       0.77      0.42      0.54       510

avg / total       0.92      0.93      0.92      5244

************************************************************
************************************************************
Category Name: fire
Accuracy: 0.99
Classification Report
             precision    recall  f1-score   support

          0       0.99      1.00      0.99      5187
          1       1.00      0.04      0.07        57

avg / total       0.99      0.99      0.98      5244

************************************************************
************************************************************
Category Name: earthquake
Accuracy: 0.96
Classification Report
             precision    recall  f1-score   support

          0       0.97      0.99      0.98      4746
          1       0.89      0.71      0.79       498

avg / total       0.96      0.96      0.96      5244

************************************************************
************************************************************
Category Name: cold
Accuracy: 0.98
Classification Report
             precision    recall  f1-score   support

          0       0.98      1.00      0.99      5132
          1       0.67      0.11      0.18       112

avg / total       0.97      0.98      0.97      5244

************************************************************
************************************************************
Category Name: other_weather
Accuracy: 0.95
Classification Report
             precision    recall  f1-score   support

          0       0.95      1.00      0.97      4957
          1       0.76      0.06      0.10       287

avg / total       0.94      0.95      0.93      5244

************************************************************
************************************************************
Category Name: direct_report
Accuracy: 0.85
Classification Report
             precision    recall  f1-score   support

          0       0.86      0.97      0.91      4240
          1       0.76      0.34      0.47      1004

avg / total       0.84      0.85      0.83      5244

************************************************************

## Acknowledgements and Data Sources:
- [Figure Eight](https://www.figure-eight.com/)

## License

MIT License

Copyright (c) 2018 Uirá Caiado

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. a summary of the results of the analysis, and necessary acknowledgements.

