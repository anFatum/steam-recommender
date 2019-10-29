from __future__ import annotations

from typing import Optional
import pandas as pd
import os
import config as cfg
from utils import convert_data_to_rating
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz


class DataMeta(type):
    _instance: Optional[SteamData] = None

    def __call__(cls) -> SteamData:
        if cls._instance is None:
            cls._instance = super(DataMeta, cls).__call__()
        return cls._instance


class SteamData(metaclass=DataMeta):
    _data: pd.DataFrame = None

    def __init__(self):
        if os.path.exists(os.sep.join([cfg.DATA_PATH, 'steam-processed.csv'])):
            self._data = pd.read_csv(os.sep.join([cfg.DATA_PATH, 'steam-processed.csv']))
        else:
            data = pd.read_csv(os.sep.join([cfg.DATA_PATH, 'steam-200k.csv']), header=None)
            self._data = self.process_data(data)
            self._data.to_csv(os.sep.join([cfg.DATA_PATH, 'steam-processed.csv']))

    def predict_games(self, game):
        game_features, game_features_matrix = self.get_features_matrix(index='game_title', columns='user_id')
        matches = []
        all_games = game_features.index.values
        for c_game in all_games:
            ratio = fuzz.ratio(game.lower(), c_game.lower())
            if ratio >= 60:
                matches.append((c_game, ratio))
        match_tuple = sorted(matches, key=lambda x: x[1])[::-1]
        if not match_tuple:
            print('Oops! No match is found')
        else:
            print('Found possible matches in our database: '
                  '{0}\n'.format([x[0] for x in match_tuple]))
            match = match_tuple[0][0]
        predictions = self.predict(game_features, game_features_matrix, match)
        print(f"Predictions for {match}")
        for i, (idx, dist) in enumerate(predictions):
            print('{0}: {1}, with distance '
                  'of {2}'.format(i + 1, game_features.index[idx], dist))

    def predict_user(self, user_id):
        game_features, game_features_matrix = self.get_features_matrix(index='user_id', columns='game_title')
        predictions = self.predict(game_features, game_features_matrix, user_id)
        for i, (idx, dist) in enumerate(predictions):
            print('{0}: {1}, with distance '
                  'of {2}'.format(i + 1, game_features.index[idx], dist))

    def predict(self, game_features, game_features_matrix, key):
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        model_knn.fit(game_features_matrix)
        n_recommendations = 10
        idx = game_features.index.get_loc(key)
        distances, indices = model_knn.kneighbors(
            game_features_matrix[idx],
            n_neighbors=n_recommendations + 1)
        raw_recommends = \
            sorted(
                list(
                    zip(
                        indices.squeeze().tolist(),
                        distances.squeeze().tolist()
                    )
                ),
                key=lambda x: x[1]
            )[:0:-1]
        return raw_recommends

    def get_features_matrix(self, index, columns):
        game_features = self._data.pivot(
            index=index,
            columns=columns,
            values='ratings'
        ).fillna(0)

        game_features_matrix = csr_matrix(game_features.values)
        return game_features, game_features_matrix

    def append_data(self, new_data: pd.DataFrame):
        processed_data = self.process_data(new_data)
        _data = self._data.append(processed_data)
        self._data.to_csv(os.sep.join([cfg.DATA_PATH, 'steam-processed.csv']))

    @classmethod
    def process_data(cls, data):
        data.columns = ["user_id", "game_title", "behavior_name", "value", "x"]

        # Dropping last column, there is no useful info
        data = data.drop(['x'], axis=1)

        # Dropping duplicates
        data = data.drop_duplicates()

        # Covert games purchased to ranking 1
        purchased = data[data['behavior_name'] == 'purchase']
        purchased.loc[:, ['value']] = 1

        # Convert games played to ranking scale
        played = data[data['behavior_name'] != 'purchase']
        played.loc[:, ['value']] = played['value'].map(convert_data_to_rating)
        played['value'].append(purchased['value']).sort_index()

        # Appending new ranking column to our data
        data_with_ranking = data.copy()
        data_with_ranking['ratings'] = played['value'].append(purchased['value']).sort_index()

        # Sort data with rankings pairwise (user-game-('purchase/play')
        sorted_rankings = data_with_ranking.sort_values(by=['user_id', 'game_title', 'behavior_name'])

        # Selecting games which was either played and purchased
        games_play_purchase = sorted_rankings[sorted_rankings.loc[:, ['user_id', 'game_title']].duplicated()]

        # Drop rows with purchased games, if they have been played
        final_data = pd.concat([data_with_ranking, games_play_purchase]).drop_duplicates(keep=False)
        final_data = final_data.drop(['behavior_name', 'value'], axis=1)

        return final_data
