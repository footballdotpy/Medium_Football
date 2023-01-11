from tabulate import tabulate
import requests
import pandas as pd
from bs4 import BeautifulSoup
import warnings
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

url = "https://www.betexplorer.com/soccer/england/premier-league-2017-2018/results/"
soup = BeautifulSoup(requests.get(url).content, "html.parser")


def get_odd_or_text(td):
    if "data-odd" in td.attrs:
        return td["data-odd"]

    odd = td.select_one("[data-odd]")
    if odd:
        return odd["data-odd"]

    return td.get_text(strip=True)


all_data = []
for row in soup.select(".table-main tr:has(td)"):
    tds = [get_odd_or_text(td) for td in row.select("td")]
    round_ = row.find_previous("th").find_previous("tr").th.text
    all_data.append([round_, *tds])

df = pd.DataFrame(
    all_data, columns=["Round", "Match", "Score", "1", "X", "2", "Date"])


df['Home'] = [i.split('-')[0] for i in df['Match']]
df['Away'] = [i.split('-')[1] for i in df['Match']]

#reverse df
df = df.iloc[::-1]


cols = ['1','X','2']

df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
df['HomeWin_Prob'] = round(1/df['1'],2)
df['DrawWin_Prob'] = round(1/df['X'],2)
df['AwayWin_Prob'] = round(1/df['2'],2)

df['HomeGoals'] = [i.split(':', 1)[0] for i in df['Score']]
df['AwayGoals'] = [i.split(':', 1)[1] for i in df['Score']]


cols = ['1','X','2']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
df['HomeWin_Prob'] = round(1/df['1'],2)
df['DrawWin_Prob'] = round(1/df['X'],2)
df['AwayWin_Prob'] = round(1/df['2'],2)

df['HomeGoals'] = [i.split(':', 1)[0] for i in df['Score']]
df['AwayGoals'] = [i.split(':', 1)[1] for i in df['Score']]


def Result(df):

    if df['HomeGoals'] > df['AwayGoals']:
       return 1
    if df['HomeGoals'] == df['AwayGoals']:
       return 0
    if df['HomeGoals'] < df['AwayGoals']:
       return 2


df['Result'] = df.apply(Result,axis=1)

df.drop(['Round','Score','Date','1','X','2','Match','HomeGoals','AwayGoals'],axis=1,inplace=True)


hold_out = df[360::]
hold_out.drop(['Result'],axis=1,inplace=True)
hold_out = hold_out.reset_index(drop=True)
df = df[0:360]


model_recode = {'Burnley':0,
              'Crystal Palace':1,
              'Huddersfield':2,
              'Liverpool':3,
              'Manchester Utd':4,
              'Newcastle':5,
              'Southampton':6,
              'Swansea':7,
              'Tottenham':8,
              'West Ham':9,
              'Chelsea':10,
              'Leicester':11,
              'Manchester City':12,
              'Arsenal':13,
              'Bournemouth':14,
              'Everton':15,
              'Stoke':16,
              'Watford':17,
              'West Brom':18,
              'Brighton':19}




# iterate over columns
for key, value in df['Home'].iteritems():
    df['Home'] = df['Home'].apply(lambda x: model_recode.get(x,x))

for key, value in df['Away'].iteritems():
    df['Away'] = df['Away'].apply(lambda x: model_recode.get(x,x))

for key, value in hold_out['Home'].iteritems():
    hold_out['Home'] = hold_out['Home'].apply(lambda x: model_recode.get(x,x))

for key, value in hold_out['Away'].iteritems():
    hold_out['Away'] = hold_out['Away'].apply(lambda x: model_recode.get(x,x))

X = df.drop('Result',axis=1)
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# use ensemble Random Forest method to improve accuracy. First we must perform GridSearchCV to find out the best params.
def model_tuning_GS(model, parameter_dict):
    """Function to perform hyperparameter turning for the classification models using GridSearch."""
    # inspect the model params.
    model.get_params()
    # define the parameters using a dictionary that we want to test.
    model_grid = parameter_dict
    # initialise a GSCV object with the model as an argument. scoring is set to accuracy and CV set to 10.
    Grid_model = GridSearchCV(estimator=model, param_grid=model_grid, cv=10, scoring="accuracy")
    # fit the model to data.
    Grid_model.fit(X_train, y_train)
    # extract the best estimator, accuracy score and print them.
    print("GridSearchCV results:", model.__class__.__name__)
    # print best estimator
    print("Best Estimator:\n", Grid_model.best_estimator_)
    # printing the mean cross-validated score of the best_estimator:
    print("\n Best Score:\n", Grid_model.best_score_)
    # printing the parameter setting that gave the best results on the hold out testing data.:
    print("\n Best Hyperparameters:\n", Grid_model.best_params_)


#  call the GridSearchCV function on the random forest.


parameter_dict = {'n_estimators': [1, 2, 3, 4, 5],
                  'max_depth': [1, 2, 3, 5, 8],
                  'min_samples_leaf': [0.1, 0.2],
                  'criterion':['gini', 'entropy', 'log_loss']}

model_tuning_GS(RandomForestClassifier(random_state=42), parameter_dict)




clf = RandomForestClassifier(criterion='gini', max_depth=5, min_samples_leaf=0.1,
                       n_estimators=5, random_state=42)

clf.fit(X_train, y_train)

# use random forest to make predictions
y_pred = clf.predict(X_test)
# print the accuracy
print("Accuracy:", metrics.accuracy_score(y_train, clf.predict(X_train)).round(decimals=4))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred).round(decimals=4))


# Make predictions on hold out set.
predictions = clf.predict(hold_out)

predictions = pd.DataFrame(predictions,columns=['Predicted_Result'])

predictions = pd.concat([predictions,hold_out],axis=1)



# reverse the team name mappings.

inv_map = {v: k for k, v in model_recode.items()}

# iterate over columns
for key, value in predictions['Home'].iteritems():
    predictions['Home'] = predictions['Home'].apply(lambda x: inv_map.get(x,x))

for key, value in predictions['Away'].iteritems():
    predictions['Away'] = predictions['Away'].apply(lambda x: inv_map.get(x,x))

print(tabulate(predictions,headers='keys'))

predictions.to_csv('pred.csv')