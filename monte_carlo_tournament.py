#!/usr/bin/env python3
"""
Monte Carlo Tournament Simulation
Simulates the Grand Swiss tournament many times to determine win probabilities
based on the current state after 4 rounds.
"""

import sys
import os
import random
import copy
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
sys.path.append('/home/geckods/Downloads/swissdutch-0.1.0')

from swissdutch.dutch import DutchPairingEngine
from swissdutch.constants import FideTitle, Colour, FloatStatus
from swissdutch.player import Player

from chess_tournament_parser import ChessTournamentParser
from elo_predictor import EloPredictor

class MonteCarloTournamentSimulator:
    """
    Monte Carlo simulation of the Grand Swiss tournament.
    Runs multiple simulations to determine win probabilities.
    """
    
    def __init__(self, num_simulations=1000, draw_probability_multiplier=1.0, tiebreak_system='aro_c1', total_rounds=11):
        self.num_simulations = num_simulations
        self.elo_predictor = EloPredictor(draw_probability_multiplier)
        self.base_players = []
        self.winner_counts = Counter()
        self.top_3_counts = Counter()
        self.final_scores = defaultdict(list)
        self.simulation_results = []
        self.num_loaded_rounds = 0  # Track how many rounds of data we have
        self.tiebreak_system = tiebreak_system
        self.total_rounds = total_rounds
        
    def load_tournament_data(self, filenames):
        """Load tournament data from multiple rounds"""
        parser = ChessTournamentParser()
        parser.parse_multiple_rounds(filenames)
        tournament_players = parser.create_player_objects()
        
        # Convert to Swiss Dutch format and store as base
        self.base_players = self._convert_to_swissdutch_players(tournament_players)
        
        # Calculate number of rounds from filenames
        self.num_loaded_rounds = len(filenames)
        print(f"Loaded {len(self.base_players)} players with results from rounds 1-{self.num_loaded_rounds}")
        return self.base_players
    
    def _convert_to_swissdutch_players(self, tournament_players):
        """Convert tournament players to Swiss Dutch format"""
        swissdutch_players = []
        
        title_mapping = {
            'GM': FideTitle.GM,
            'IM': FideTitle.IM,
            'FM': FideTitle.FM,
            'CM': FideTitle.CM,
            'WGM': FideTitle.WGM,
            'WIM': FideTitle.WIM,
            'WFM': FideTitle.WFM,
            'WCM': FideTitle.WCM
        }
        
        for player in tournament_players:
            # Convert color history
            colour_hist = []
            for colour in player.colour_hist:
                if hasattr(colour, 'value'):
                    if colour.value == 1:
                        colour_hist.append(Colour.white)
                    elif colour.value == -1:
                        colour_hist.append(Colour.black)
                    else:
                        colour_hist.append(Colour.none)
                else:
                    colour_hist.append(Colour.none)
            
            swissdutch_player = Player(
                name=player.name,
                rating=player.rating,
                title=title_mapping.get(player.title, None),
                pairing_no=player.pairing_no,
                score=player.score,
                float_status=FloatStatus.none,
                opponents=player.opponents,
                colour_hist=tuple(colour_hist)
            )
            swissdutch_players.append(swissdutch_player)
        
        return swissdutch_players
    
    def create_fresh_simulation_copy(self):
        """Create a fresh copy of players for each simulation"""
        return copy.deepcopy(self.base_players)
    
    def calculate_aro_c1(self, player, all_players):
        """Calculate Average Rating of Opponents Cut 1 (ARO-C1)"""
        if not player.opponents:
            return 0.0
        
        # Get opponent ratings, excluding byes
        opponent_ratings = []
        for opp_no in player.opponents:
            # Find opponent player
            opp_player = next((p for p in all_players if p.pairing_no == opp_no), None)
            if opp_player:
                opponent_ratings.append(opp_player.rating)
        
        if not opponent_ratings:
            return 0.0
        
        # Sort ratings and cut the lowest one (Cut 1)
        sorted_ratings = sorted(opponent_ratings)
        if len(sorted_ratings) > 1:
            # Remove the lowest rating
            ratings_to_average = sorted_ratings[1:]
        else:
            # If only one opponent, use that rating
            ratings_to_average = sorted_ratings
        
        return sum(ratings_to_average) / len(ratings_to_average) if ratings_to_average else 0.0

    def get_sorting_key(self, player, players):
        """Get sorting key based on tiebreak system"""
        if self.tiebreak_system == 'aro_c1':
            return (player.score, self.calculate_aro_c1(player, players), player.rating)
        elif self.tiebreak_system == 'rating_only':
            return (player.score, player.rating)
        else:
            # Default to ARO-C1
            return (player.score, self.calculate_aro_c1(player, players), player.rating)

    def pair_round_simple(self, players, round_num):
        """Simple pairing algorithm for remaining rounds"""
        # Sort players based on selected tiebreak system
        sorted_players = sorted(players, key=lambda p: self.get_sorting_key(p, players), reverse=True)
        
        # Simple pairing: pair players with similar scores
        pairings = []
        used_players = set()
        
        for i, player_a in enumerate(sorted_players):
            if player_a.pairing_no in used_players:
                continue
                
            # Find best opponent with similar score who hasn't been paired
            best_opponent = None
            best_score_diff = float('inf')
            
            for j, player_b in enumerate(sorted_players[i+1:], i+1):
                if (player_b.pairing_no in used_players or 
                    player_b.pairing_no in player_a.opponents):
                    continue
                
                # Prefer players with similar scores
                score_diff = abs(player_a.score - player_b.score)
                if score_diff < best_score_diff:
                    best_score_diff = score_diff
                    best_opponent = player_b
            
            if best_opponent:
                pairings.append((player_a, best_opponent))
                used_players.add(player_a.pairing_no)
                used_players.add(best_opponent.pairing_no)
            else:
                # Fallback: find any unpaired opponent (avoid infinite loops)
                for player_b in sorted_players[i+1:]:
                    if (player_b.pairing_no not in used_players and 
                        player_b.pairing_no not in player_a.opponents):
                        pairings.append((player_a, player_b))
                        used_players.add(player_a.pairing_no)
                        used_players.add(player_b.pairing_no)
                        break
        
        # Update player information
        for player_a, player_b in pairings:
            # Determine colors (alternate based on previous games)
            if len(player_a.colour_hist) > 0 and player_a.colour_hist[-1] == Colour.white:
                player_a_colour = Colour.black
                player_b_colour = Colour.white
            else:
                player_a_colour = Colour.white
                player_b_colour = Colour.black
            
            # Update opponents and color history
            player_a._opponents = player_a.opponents + (player_b.pairing_no,)
            player_b._opponents = player_b.opponents + (player_a.pairing_no,)
            player_a._colour_hist = player_a.colour_hist + (player_a_colour,)
            player_b._colour_hist = player_b.colour_hist + (player_b_colour,)
        
        return pairings
    
    def simulate_round_results(self, pairings):
        """Simulate game results for the current round using Elo probabilities"""
        for player_a, player_b in pairings:
            # Get Elo predictions
            probs_a, probs_b = self.elo_predictor.calculate_matchup_probabilities(
                player_a.rating, player_b.rating
            )
            
            # Simulate result based on probabilities
            rand = random.random()
            
            if rand < probs_a.win_prob:
                # Player A wins
                player_a._score += 1
                player_b._score += 0
            elif rand < probs_a.win_prob + probs_a.draw_prob:
                # Draw
                player_a._score += 0.5
                player_b._score += 0.5
            else:
                # Player B wins
                player_a._score += 0
                player_b._score += 1
    
    def run_single_simulation(self):
        """Run a single complete tournament simulation"""
        players = self.create_fresh_simulation_copy()
        
        # Run remaining rounds (from loaded_rounds + 1 to total_rounds)
        start_round = self.num_loaded_rounds + 1
        for round_num in range(start_round, self.total_rounds + 1):
            # Pair the round
            pairings = self.pair_round_simple(players, round_num)
            
            # Simulate results
            self.simulate_round_results(pairings)
        
        # Determine winner and top 3
        sorted_players = sorted(players, key=lambda p: (p.score, p.rating), reverse=True)
        winner = sorted_players[0]
        top_3 = [p.name for p in sorted_players[:3]]
        
        return {
            'winner': winner.name,
            'winner_rating': winner.rating,
            'winner_score': winner.score,
            'top_3': top_3,
            'final_standings': [(p.name, p.score, p.rating) for p in sorted_players]
        }
    
    def run_monte_carlo_simulation(self):
        """Run Monte Carlo simulation with multiple tournament runs"""
        print(f"Running {self.num_simulations} tournament simulations...")
        print("This may take a few minutes...")
        
        for i in tqdm(range(self.num_simulations), desc="Simulating tournaments"):
            # Set different random seed for each simulation
            random.seed(i)
            
            result = self.run_single_simulation()
            self.simulation_results.append(result)
            
            # Track statistics
            self.winner_counts[result['winner']] += 1
            for player in result['top_3']:
                self.top_3_counts[player] += 1
            
            # Track final scores for each player
            for name, score, rating in result['final_standings']:
                self.final_scores[name].append(score)
        
        print(f"Completed {self.num_simulations} simulations!")
        return self.simulation_results
    
    def analyze_results(self):
        """Analyze and present Monte Carlo results"""
        print("\n" + "="*80)
        print("MONTE CARLO TOURNAMENT SIMULATION RESULTS")
        print("="*80)
        
        # Winner probabilities
        print(f"\n🏆 WINNER PROBABILITIES (Top 15):")
        print(f"{'Rank':<4} {'Player':<25} {'Wins':<6} {'Probability':<12} {'Rating':<7}")
        print("-" * 65)
        
        total_simulations = sum(self.winner_counts.values())
        for i, (player, wins) in enumerate(self.winner_counts.most_common(15), 1):
            probability = (wins / total_simulations) * 100
            # Find player rating
            player_rating = next((p.rating for p in self.base_players if p.name == player), 0)
            print(f"{i:<4} {player:<25} {wins:<6} {probability:>8.2f}% {player_rating:<7}")
        
        # Top 3 probabilities
        print(f"\n🥇🥈🥉 TOP 3 FINISH PROBABILITIES (Top 15):")
        print(f"{'Rank':<4} {'Player':<25} {'Top 3s':<7} {'Probability':<12} {'Rating':<7}")
        print("-" * 65)
        
        for i, (player, top_3s) in enumerate(self.top_3_counts.most_common(15), 1):
            probability = (top_3s / total_simulations) * 100
            player_rating = next((p.rating for p in self.base_players if p.name == player), 0)
            print(f"{i:<4} {player:<25} {top_3s:<7} {probability:>8.2f}% {player_rating:<7}")
        
        # Expected final scores
        print(f"\n📊 EXPECTED FINAL SCORES (Top 15):")
        print(f"{'Rank':<4} {'Player':<25} {'Avg Score':<10} {'Std Dev':<8} {'Rating':<7}")
        print("-" * 65)
        
        avg_scores = []
        for player in self.base_players:
            if player.name in self.final_scores:
                scores = self.final_scores[player.name]
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                avg_scores.append((player.name, avg_score, std_score, player.rating))
        
        avg_scores.sort(key=lambda x: x[1], reverse=True)
        for i, (name, avg, std, rating) in enumerate(avg_scores[:15], 1):
            print(f"{i:<4} {name:<25} {avg:>7.2f} {std:>6.2f} {rating:<7}")
        
        return {
            'winner_probs': dict(self.winner_counts.most_common()),
            'top3_probs': dict(self.top_3_counts.most_common()),
            'expected_scores': {name: np.mean(scores) for name, scores in self.final_scores.items()}
        }
    
    def create_histogram(self, save_path="tournament_winner_histogram.png"):
        """Create histogram of tournament winners"""
        if not self.winner_counts:
            print("No simulation results available. Run simulation first.")
            return
        
        # Get top 20 winners for cleaner visualization
        top_winners = dict(self.winner_counts.most_common(20))
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Create histogram
        players = list(top_winners.keys())
        wins = list(top_winners.values())
        probabilities = [(w / sum(self.winner_counts.values())) * 100 for w in wins]
        
        # Create bar plot
        bars = plt.bar(range(len(players)), probabilities, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Customize plot
        plt.xlabel('Players', fontsize=12)
        plt.ylabel('Win Probability (%)', fontsize=12)
        plt.title(f'Monte Carlo Tournament Simulation Results\n({self.num_simulations} simulations)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(players)), players, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{prob:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Add rating information
        rating_text = []
        for player in players:
            rating = next((p.rating for p in self.base_players if p.name == player), 0)
            rating_text.append(f'{rating}')
        
        # Add rating labels below player names
        ax2 = plt.gca().twiny()
        ax2.set_xlim(plt.gca().get_xlim())
        ax2.set_xticks(range(len(players)))
        ax2.set_xticklabels(rating_text, rotation=45, ha='right', fontsize=8, color='red')
        ax2.set_xlabel('Player Ratings', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Histogram saved as: {save_path}")
    
    def create_top3_histogram(self, save_path="tournament_top3_histogram.png"):
        """Create histogram of top 3 finish probabilities"""
        if not self.top_3_counts:
            print("No simulation results available. Run simulation first.")
            return
        
        # Get top 20 for cleaner visualization
        top_players = dict(self.top_3_counts.most_common(20))
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Create histogram
        players = list(top_players.keys())
        top3s = list(top_players.values())
        probabilities = [(t / sum(self.winner_counts.values())) * 100 for t in top3s]
        
        # Create bar plot
        bars = plt.bar(range(len(players)), probabilities, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        
        # Customize plot
        plt.xlabel('Players', fontsize=12)
        plt.ylabel('Top 3 Finish Probability (%)', fontsize=12)
        plt.title(f'Monte Carlo Tournament Simulation - Top 3 Finish Probabilities\n({self.num_simulations} simulations)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(players)), players, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{prob:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Top 3 histogram saved as: {save_path}")

def main():
    """Run Monte Carlo tournament simulation"""
    # Set up simulation
    num_simulations = 1000  # Adjust this number as needed
    simulator = MonteCarloTournamentSimulator(num_simulations)
    
    # Load tournament data
    filenames = [
        'Chess Data - Round 1.csv',
        'Chess Data - Round 2.csv',
        'Chess Data - Round 3.csv',
        'Chess Data - Round 4.csv'
    ]
    
    simulator.load_tournament_data(filenames)
    
    # Show current standings after 4 rounds
    print("\n=== Current Standings After Round 4 ===")
    sorted_players = sorted(simulator.base_players, key=lambda p: (p.score, p.rating), reverse=True)
    for i, player in enumerate(sorted_players[:10], 1):
        print(f"{i:2d}. {player.name:<25} | Rating: {player.rating:4d} | Score: {player.score:3.1f}")
    
    # Run Monte Carlo simulation
    simulator.run_monte_carlo_simulation()
    
    # Analyze results
    results = simulator.analyze_results()
    
    # Create histograms
    simulator.create_histogram()
    simulator.create_top3_histogram()
    
    return results

if __name__ == "__main__":
    # Run the Monte Carlo simulation
    results = main()
