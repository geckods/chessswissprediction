import csv
from collections import defaultdict
from swissdutch.player import Player
from swissdutch.constants import FloatStatus, Colour

class ChessTournamentParser:
    def __init__(self):
        self.players = {}
        self.games = []
    
    def parse_round_file(self, filename):
        """Parse a single round CSV file and extract game data"""
        with open(filename, 'r', encoding='utf-8') as file:
            # Read the raw data to handle duplicate column names
            lines = file.readlines()
            header = lines[0].strip().split(',')
            
            # Create a proper reader with unique column names
            import io
            # Rename duplicate columns
            unique_headers = []
            for i, h in enumerate(header):
                if h in unique_headers:
                    unique_headers.append(f"{h}_{i}")
                else:
                    unique_headers.append(h)
            
            # Create new CSV content with unique headers
            new_content = ','.join(unique_headers) + '\n' + ''.join(lines[1:])
            reader = csv.DictReader(io.StringIO(new_content))
            
            for row in reader:
                # Extract player 1 (left side - white)
                p1_sno = int(row['SNo.'])
                p1_name = row['NAME'].strip('"')
                p1_title = row['Title']
                p1_rating = int(row['Rtg'])
                p1_score = float(row['Pts'])
                
                # Extract player 2 (right side - black)
                p2_sno = int(row['SNo._11'])  # The second SNo. column
                p2_name = row['NAME_9'].strip('"')  # The second NAME column
                p2_title = row['Title_8']  # The second Title column
                p2_rating = int(row['Rtg_10'])  # The second Rtg column
                p2_score = float(row['Pts_7'])  # The second Pts column
                
                # Extract game result
                result = row['Res']
                
                # Store player information
                if p1_sno not in self.players:
                    self.players[p1_sno] = {
                        'name': p1_name,
                        'title': p1_title,
                        'rating': p1_rating,
                        'pairing_no': p1_sno,
                        'games': []
                    }
                
                if p2_sno not in self.players:
                    self.players[p2_sno] = {
                        'name': p2_name,
                        'title': p2_title,
                        'rating': p2_rating,
                        'pairing_no': p2_sno,
                        'games': []
                    }
                
                # Store game information
                game = {
                    'white_player': p1_sno,
                    'black_player': p2_sno,
                    'result': result,
                    'white_score': p1_score,
                    'black_score': p2_score
                }
                self.games.append(game)
                
                # Add game to player records
                self.players[p1_sno]['games'].append({
                    'opponent': p2_sno,
                    'colour': Colour.white,
                    'result': result
                })
                self.players[p2_sno]['games'].append({
                    'opponent': p1_sno,
                    'colour': Colour.black,
                    'result': result
                })
    
    def parse_multiple_rounds(self, filenames):
        """Parse multiple round files"""
        for filename in filenames:
            self.parse_round_file(filename)
    
    def create_player_objects(self):
        """Create Player objects from parsed data"""
        player_objects = []
        
        for sno, player_data in self.players.items():
            # Calculate total score from all games
            total_score = 0
            opponents = []
            colour_history = []
            
            for game in player_data['games']:
                opponents.append(game['opponent'])
                colour_history.append(game['colour'])
                
                # Calculate score for this game
                result = game['result']
                if result == '1 - 0':  # White wins
                    if game['colour'] == Colour.white:
                        total_score += 1
                    else:
                        total_score += 0
                elif result == '0 - 1':  # Black wins
                    if game['colour'] == Colour.black:
                        total_score += 1
                    else:
                        total_score += 0
                elif result == '½ - ½':  # Draw
                    total_score += 0.5
            
            # Create Player object
            player = Player(
                name=player_data['name'],
                rating=player_data['rating'],
                title=player_data['title'],
                pairing_no=player_data['pairing_no'],
                score=total_score,
                float_status=FloatStatus.none,  # You may want to calculate this based on your logic
                opponents=tuple(opponents),
                colour_hist=tuple(colour_history)
            )
            
            player_objects.append(player)
        
        return player_objects
    
    def get_player_by_sno(self, sno):
        """Get a specific player by their SNo"""
        if sno in self.players:
            player_data = self.players[sno]
            # Calculate score, opponents, and colour history
            total_score = 0
            opponents = []
            colour_history = []
            
            for game in player_data['games']:
                opponents.append(game['opponent'])
                colour_history.append(game['colour'])
                
                # Calculate score for this game
                result = game['result']
                if result == '1 - 0':  # White wins
                    if game['colour'] == Colour.white:
                        total_score += 1
                    else:
                        total_score += 0
                elif result == '0 - 1':  # Black wins
                    if game['colour'] == Colour.black:
                        total_score += 1
                    else:
                        total_score += 0
                elif result == '½ - ½':  # Draw
                    total_score += 0.5
            
            return Player(
                name=player_data['name'],
                rating=player_data['rating'],
                title=player_data['title'],
                pairing_no=player_data['pairing_no'],
                score=total_score,
                float_status=FloatStatus.none,
                opponents=tuple(opponents),
                colour_hist=tuple(colour_history)
            )
        return None

def main():
    """Example usage"""
    parser = ChessTournamentParser()
    
    # Parse all rounds
    filenames = [
        '/home/geckods/Coding/chessswissprediction/Chess Data - Round 1.csv',
        '/home/geckods/Coding/chessswissprediction/Chess Data - Round 2.csv',
        '/home/geckods/Coding/chessswissprediction/Chess Data - Round 3.csv',
        '/home/geckods/Coding/chessswissprediction/Chess Data - Round 4.csv'
    ]
    
    parser.parse_multiple_rounds(filenames)
    
    # Create all player objects
    players = parser.create_player_objects()
    
    # Print first few players as examples
    print("Sample Player Objects:")
    for i, player in enumerate(players[:5]):
        print(f"Player {i+1}:")
        print(f"  Name: {player.name}")
        print(f"  Rating: {player.rating}")
        print(f"  Title: {player.title}")
        print(f"  Pairing No: {player.pairing_no}")
        print(f"  Score: {player.score}")
        print(f"  Opponents: {player.opponents}")
        print(f"  Colour History: {player.colour_hist}")
        print()
    
    # Example: Get a specific player
    pragg_player = parser.get_player_by_sno(1)  # Praggnanandhaa
    if pragg_player:
        print("Praggnanandhaa's Player Object:")
        print(f"Player(name='{pragg_player.name}',")
        print(f"       rating={pragg_player.rating},")
        print(f"       title='{pragg_player.title}',")
        print(f"       pairing_no={pragg_player.pairing_no},")
        print(f"       score={pragg_player.score},")
        print(f"       float_status=FloatStatus.none,")
        print(f"       opponents={pragg_player.opponents},")
        print(f"       colour_hist={pragg_player.colour_hist})")

if __name__ == "__main__":
    main()
