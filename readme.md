# IPL Victory Estimator

This project aims to estimate the probability of a team winning an IPL match using historical match data and machine learning techniques. The model is built using a Logistic Regression algorithm after performing necessary data preprocessing and feature engineering.

## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
- [Usage](#usage)
- [License](#license)

## Installation

### Step 1: Clone the repository
Clone the project repository to your local machine:

```bash
git clone https://github.com/yourusername/IPL-Victory-Estimator.git
cd IPL-Victory-Estimator
```

### Step 2: Set up a virtual environment

Create a virtual environment to isolate project dependencies:

```bash
python -m venv .venv
```

### Step 3: Activate the virtual environment

Activate the virtual environment:

- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### Step 4: Install dependencies

Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

### Step 5: Deactivate the virtual environment (Optional)

When you're done, you can deactivate the virtual environment:

```bash
deactivate
```

## Data Preparation

The project uses two datasets: `matches.csv` and `deliveries.csv`.

### Loading the Data

```python
import pandas as pd

match_df = pd.read_csv('Data/matches.csv')
deliveries_df = pd.read_csv('Data/deliveries.csv')
```

### Data Cleaning

We remove unnecessary columns and rename some for clarity:

```python
match_df.drop(['toss_winner','toss_decision','player_of_match','umpire1', 'umpire2','umpire3'], axis=1, inplace=True)
deliveries_df.drop(['bowler','is_super_over','dismissal_kind','fielder','wide_runs','bye_runs','legbye_runs','noball_runs','penalty_runs'], axis=1, inplace=True)
```

### Feature Engineering

#### Calculating Total Score

We calculate the total runs scored in the first innings:

```python
total_scored_df = (deliveries_df.groupby(['match_id','inning']).sum()['total_runs'].reset_index())
total_scored_df = total_scored_df[total_scored_df['inning'] == 1]
match_df = match_df.merge(total_scored_df[['match_id','total_runs']], left_on='id', right_on='match_id')
```

#### Team Name Corrections

Some team names in the dataset need to be corrected to their current names:

```python
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['team1'] = match_df['team1'].str.replace('Gujarat Lions', 'Gujarat Titans')
match_df['team2'] = match_df['team2'].str.replace('Gujarat Lions', 'Gujarat Titans')
```

#### Filtering Matches

We remove matches where the Duckworth-Lewis (DLS) method was applied:

```python
match_df = match_df[match_df['dl_applied'] == 0]
```

#### Calculating Remaining Metrics

We calculate additional metrics such as the target left, remaining balls, and current run rate (CRR):

```python
deliveries_df['score'] = deliveries_df[['match_id', 'Ball_score']].groupby('match_id').cumsum()['Ball_score']
deliveries_df['target_left'] = (deliveries_df['total_runs'] + 1) - deliveries_df['score']
deliveries_df['Remaining Balls'] = (120 - ((deliveries_df['over'] - 1) * 6  + deliveries_df['ball']))
deliveries_df['player_dismissed'] = deliveries_df['player_dismissed'].apply(lambda x:x if x == '0' else '1').astype('int64')
deliveries_df['wickets'] = deliveries_df[['match_id','player_dismissed']].groupby('match_id').cumsum()['player_dismissed'].values
deliveries_df['wickets'] = 10 - deliveries_df['wickets']
deliveries_df['CRR'] = (deliveries_df['score']) * 6 / (120 - deliveries_df['Remaining Balls'])
deliveries_df['RRR'] = (deliveries_df['target_left']) * 6 / (deliveries_df['Remaining Balls'])
```

## Model Building

### Preparing the Dataset

We split the data into features (`X`) and target (`Y`):

```python
model_df = deliveries_df[['batting_team','bowling_team','city','score', 'wickets', 'Remaining Balls', 'target_left', 'CRR', 'RRR', 'result']]
X = model_df.iloc[:, :-1]
Y = model_df.iloc[:, -1]
```

### Encoding Categorical Features

We apply OneHotEncoding to transform categorical features into numerical format:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')
```

### Building and Training the Model

We use a Logistic Regression model to estimate the win probability:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', LogisticRegression(solver='liblinear'))
])

pipe.fit(X_train, Y_train)
Y_prediction = pipe.predict(X_test)
```

## Usage

To use the project, run the following command in your terminal after setting up the environment:

```bash
streamlit run app.py
```

This will start a Streamlit app that provides an interface for estimating IPL match outcomes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This `README.md` should provide clear guidance on how to set up and use your "IPL Victory Estimator" project, including data preparation, feature engineering, and model building. Feel free to adjust the details according to your project's specifics!