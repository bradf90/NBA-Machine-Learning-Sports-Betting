import argparse
from datetime import datetime, timedelta
import time
from colorama import Fore, Style
import pandas as pd
import tensorflow as tf
from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame, get_todays_games_json, create_todays_games
from src.DataProviders.SbrOddsProvider import SbrOddsProvider
tensorflow==2.11.0
requests==2.28.2
tqdm==4.64.1
colorama==0.4.6
pandas==1.5.3
numpy==1.24.2
scikit-learn==1.2.1
keras==2.11.0
xgboost==1.7.4
sbrscrape==0.0.6
Flask==2.2.3
openpyxl==3.1.1

todays_games_url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2022/scores/00_todays_scores.json'
data_url = 'https://stats.nba.com/stats/leaguedashteamstats?' \
           'Conference=&DateFrom=&DateTo=&Division=&GameScope=&' \
           'GameSegment=&LastNGames=0&LeagueID=00&Location=&' \
           'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&' \
           'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&' \
           'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&' \
           'Season=2022-23&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
           'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='


def createTodaysGames(games, df, odds):
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []
    # todo: get the days rest for current games
    home_team_days_rest = []
    away_team_days_rest = []

    for game in games:
        home_team = game[0]
        away_team = game[1]
        if home_team not in team_index_current or away_team not in team_index_current:
            continue
        if odds is not None:
            game_odds = odds[home_team + ':' + away_team]
            todays_games_uo.append(game_odds['under_over_odds'])
            
            home_team_odds.append(game_odds[home_team]['money_line_odds'])
            away_team_odds.append(game_odds[away_team]['money_line_odds'])

        else:
            todays_games_uo.append(input(home_team + ' vs ' + away_team + ': '))

            home_team_odds.append(input(home_team + ' odds: '))
            away_team_odds.append(input(away_team + ' odds: '))
        
        # calculate days rest for both teams
        dateparse = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M')
        schedule_df = pd.read_csv('Data/nba-2022-UTC.csv', parse_dates=['Date'], date_parser=dateparse)
        home_games = schedule_df[(schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)]
        away_games = schedule_df[(schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)]
        last_home_date = home_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date',ascending=False).head(1)['Date'].iloc[0]
        last_away_date = away_games.loc[schedule_df['Date'] <= datetime.today()].sort_values('Date',ascending=False).head(1)['Date'].iloc[0]
        home_days_off = timedelta(days=1) + datetime.today() - last_home_date
        away_days_off = timedelta(days=1) + datetime.today() - last_away_date
        # print(f"{away_team} days off: {away_days_off.days} @ {home_team} days off: {home_days_off.days}")

        home_team_days_rest.append(home_days_off.days)
        away_team_days_rest.append(away_days_off.days)
        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)]
        stats = pd.concat([home_team_series, away_team_series])
        stats['Days-Rest-Home'] = home_days_off.days
        stats['Days-Rest-Away'] = away_days_off.days
        match_data.append(stats)

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
    data = frame_ml.values
    data = data.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds


def main():
    odds = None
    if args.odds:
        odds = SbrOddsProvider(sportsbook=args.odds).get_odds()
        games = create_todays_games_from_odds(odds)
        if len(games) == 0:
            print("No games found.")
            return
        if((games[0][0]+':'+games[0][1]) not in list(odds.keys())):
            print(games[0][0]+':'+games[0][1])
            print(Fore.RED, "--------------Games list not up to date for todays games!!! Scraping disabled until list is updated.--------------")
            print(Style.RESET_ALL)
            odds = None
        else:
            print(f"------------------{args.odds} odds data------------------")
            for g in odds.keys():
                home_team, away_team = g.split(":")
                print(f"{away_team} ({odds[g][away_team]['money_line_odds']}) @ {home_team} ({odds[g][home_team]['money_line_odds']})")
    else:
        data = get_todays_games_json(todays_games_url)
        games = create_todays_games(data)
    data = get_json_data(data_url)
    df = to_data_frame(data)
    data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = createTodaysGames(games, df, odds)
    if args.nn:
        print("------------Neural Network Model Predictions-----------")
        data = tf.keras.utils.normalize(data, axis=1)
        NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds)
        print("-------------------------------------------------------")
    if args.xgb:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds)
        print("-------------------------------------------------------")
    if args.A:
        print("---------------XGBoost Model Predictions---------------")
        XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds)
        print("-------------------------------------------------------")
        data = tf.keras.utils.normalize(data, axis=1)
        print("------------Neural Network Model Predictions-----------")
        NN_Runner.nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds)
        print("-------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-nn', action='store_true', help='Run with Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run all Models')
    parser.add_argument('-odds', help='Sportsbook to fetch from. (fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny')
    args = parser.parse_args()
    main()
import json
import selenium
import sys
from selenium import webdriver
from time import sleep


BROWSER_LOAD_DELAY = 6


conn = None
def get_connection():
    """Returns a StatsNbaComApi connection
    """
    global conn
    if conn is None:
        conn = StatsNbaComApi()
    return conn


class StatsNbaComApi:
    """
    A stats.nba.com API using selenium.
    """
    def __init__(self):
        self.browser = None

    def __del__(self):
        if self.browser:
            self.browser.close()

    def connect(self):
        """Connects to the stats.nba.com API through a browser window.
        Does not seem to work with headless browsers.
        """
        if self.browser:
            return
        options = webdriver.ChromeOptions()
        self.browser = webdriver.Chrome(options=options)
        self.browser.get('https://stats.nba.com')
        sleep(BROWSER_LOAD_DELAY)

    def get(self, uri):
        """Sends an API request to stats.nba.com
        """
        if self.browser is None:
            self.connect()
        return json.loads(self.browser.execute_script(f'''
            var req = new XMLHttpRequest();
            req.open("GET", "{uri}", false);
            req.send(null);
            return req.responseText;
        '''))


if __name__ == '__main__':
    """When run as a main, the program fetches and prints a stats.nba.com uri.
    Program takes one argument: the url to lead
    """

    ## Exit if the program was run without a single arg: the URL
    if len(sys.argv) != 2:
        print('Program takes one arg: the uri to fetch')
        sys.exit(1)
    
    ## Create an API instance
    api = StatsNbaComApi()

    ## Fetch the data
    uri = sys.argv[1]
    data = api.get(uri)
    print(data)