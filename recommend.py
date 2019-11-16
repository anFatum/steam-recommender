from SteamData import SteamData
import argparse
import sys
#
# ag = argparse.ArgumentParser()
# ag.add_argument('-g', '--game', type=str, required=False, help="Game title to predict recommendation to")
# ag.add_argument('-u', '--user', type=int, required=False, help="User id to predict recommendation to")
# ag.add_argument('-r_num', '--recommendation_number', required=False, type=int, help="Number of recommendation")

recommend = SteamData()
recommend.predict_games("Fallout 4")

# sys.stdout.flush()
