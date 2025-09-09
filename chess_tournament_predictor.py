#!/usr/bin/env python3
"""
Chess Tournament Predictor
Pure MCTS-based prediction system for Swiss chess tournaments
"""

import json
import argparse
from typing import Dict, List
from datetime import datetime
import matplotlib.pyplot as plt

from chess_tournament_parser import ChessTournamentParser
from monte_carlo_tournament import MonteCarloTournamentSimulator
from elo_predictor import EloPredictor

class ChessTournamentPredictor:
    """Main system for predicting chess tournament outcomes using MCTS"""
    
    def __init__(self, draw_probability_multiplier: float = 1.0, tiebreak_system: str = 'aro_c1', total_rounds: int = 11):
        self.parser = ChessTournamentParser()
        self.predictor = EloPredictor(draw_probability_multiplier)
        self.monte_carlo = MonteCarloTournamentSimulator(
            draw_probability_multiplier=draw_probability_multiplier,
            tiebreak_system=tiebreak_system,
            total_rounds=total_rounds
        )
        
    def load_tournament_data(self, csv_files: List[str]) -> List:
        """Load tournament data from CSV files"""
        print(f"📊 Loading tournament data from {len(csv_files)} files...")
        
        # Use the existing MonteCarloTournamentSimulator to load data
        self.monte_carlo.load_tournament_data(csv_files)
        print(f"✅ Loaded {len(self.monte_carlo.base_players)} players")
        
        return self.monte_carlo.base_players
    
    def run_simulation(self, players: List, num_simulations: int = 2000) -> Dict:
        """Run Monte Carlo simulation"""
        print(f"🎲 Running {num_simulations} Monte Carlo simulations...")
        
        # Update the number of simulations
        self.monte_carlo.num_simulations = num_simulations
        
        # Run the simulation
        self.monte_carlo.run_monte_carlo_simulation()
        
        # Get results
        results = self.monte_carlo.analyze_results()
        
        # Add missing fields to results
        results['total_simulations'] = self.monte_carlo.num_simulations
        results['winner_counts'] = self.monte_carlo.winner_counts
        
        print(f"✅ Simulation complete!")
        print(f"   Total simulations: {results['total_simulations']}")
        print(f"   Unique winners: {len(results['winner_counts'])}")
        
        return results
    
    def analyze_results(self, results: Dict) -> Dict:
        """Analyze simulation results"""
        print("📈 Analyzing results...")
        
        winner_counts = results['winner_counts']
        total_sims = results['total_simulations']
        
        # Calculate win probabilities
        win_probabilities = {}
        for player, count in winner_counts.items():
            win_probabilities[player] = count / total_sims
        
        # Sort by win probability
        sorted_players = sorted(win_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate top 3 probability
        top_3_prob = sum(prob for _, prob in sorted_players[:3])
        
        # Calculate expected scores (assuming 11 rounds, 1 point per win, 0.5 per draw)
        expected_scores = {}
        for player, prob in win_probabilities.items():
            # Rough estimate: winner gets ~8.5 points on average
            expected_scores[player] = prob * 8.5
        
        analysis = {
            'win_probabilities': win_probabilities,
            'sorted_players': sorted_players,
            'top_3_probability': top_3_prob,
            'expected_scores': expected_scores,
            'total_simulations': total_sims
        }
        
        return analysis
    
    def display_results(self, analysis: Dict):
        """Display prediction results"""
        print("\n" + "="*80)
        print("🏆 CHESS TOURNAMENT PREDICTION RESULTS")
        print("="*80)
        
        sorted_players = analysis['sorted_players']
        top_3_prob = analysis['top_3_probability']
        
        print(f"\n📊 TOP 10 WIN PROBABILITIES:")
        print(f"{'Rank':<4} {'Player':<30} {'Win %':<8} {'Expected Score':<12}")
        print("-" * 60)
        
        for i, (player, prob) in enumerate(sorted_players[:10], 1):
            expected_score = analysis['expected_scores'].get(player, 0)
            print(f"{i:<4} {player[:29]:<30} {prob*100:<7.1f}% {expected_score:<11.1f}")
        
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   Top 3 Combined Probability: {top_3_prob*100:.1f}%")
        print(f"   Total Simulations: {analysis['total_simulations']:,}")
        print(f"   Unique Winners: {len(analysis['win_probabilities'])}")
        
        # Calculate Gini coefficient for inequality
        probs = [prob for _, prob in sorted_players]
        gini = self._calculate_gini(probs)
        print(f"   Winner Inequality (Gini): {gini:.3f}")
        
        if gini > 0.7:
            print("   📊 High inequality - tournament is very top-heavy")
        elif gini > 0.5:
            print("   📊 Moderate inequality - some clear favorites")
        else:
            print("   📊 Low inequality - tournament is quite open")
    
    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values:
            return 0
        
        n = len(values)
        sorted_values = sorted(values)
        cumsum = [sum(sorted_values[:i+1]) for i in range(n)]
        
        gini = (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum, 1))) / (n * sum(sorted_values))
        return gini
    
    def save_results(self, analysis: Dict, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tournament_prediction_{timestamp}.json"
        
        # Convert to serializable format
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'total_simulations': analysis['total_simulations'],
            'win_probabilities': analysis['win_probabilities'],
            'top_3_probability': analysis['top_3_probability'],
            'expected_scores': analysis['expected_scores'],
            'sorted_players': analysis['sorted_players']
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Results saved to {filename}")
    
    def plot_results(self, analysis: Dict, filename: str = None):
        """Create visualization of results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tournament_prediction_{timestamp}.png"
        
        sorted_players = analysis['sorted_players'][:15]  # Top 15
        
        players = [p[0][:20] for p in sorted_players]  # Truncate names
        probabilities = [p[1] * 100 for p in sorted_players]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(players)), probabilities, color='steelblue', alpha=0.7)
        
        plt.xlabel('Players')
        plt.ylabel('Win Probability (%)')
        plt.title('Chess Tournament Win Probabilities (Top 15)')
        plt.xticks(range(len(players)), players, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{prob:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to {filename}")
    
    def run_complete_analysis(self, csv_files: List[str], num_simulations: int = 2000, 
                            save_results: bool = True, create_plot: bool = True,
                            draw_probability_multiplier: float = 1.0, tiebreak_system: str = 'aro_c1',
                            total_rounds: int = 11):
        """Run complete tournament prediction analysis"""
        print("🏁 Starting Chess Tournament Prediction Analysis")
        print("="*60)
        print(f"📊 Draw Probability Multiplier: {draw_probability_multiplier:.2f}")
        print(f"📊 Tiebreak System: {tiebreak_system}")
        print(f"📊 Total Rounds: {total_rounds}")
        
        # Reinitialize with the specified parameters
        self.predictor = EloPredictor(draw_probability_multiplier)
        self.monte_carlo = MonteCarloTournamentSimulator(
            draw_probability_multiplier=draw_probability_multiplier,
            tiebreak_system=tiebreak_system,
            total_rounds=total_rounds
        )
        
        # Load data
        players = self.load_tournament_data(csv_files)
        
        # Run simulation
        results = self.run_simulation(players, num_simulations)
        
        # Analyze results
        analysis = self.analyze_results(results)
        
        # Display results
        self.display_results(analysis)
        
        # Save results
        if save_results:
            self.save_results(analysis)
        
        # Create plot
        if create_plot:
            self.plot_results(analysis)
        
        print("\n✅ Analysis complete!")
        return analysis

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Chess Tournament Predictor')
    parser.add_argument('--simulations', type=int, default=2000,
                       help='Number of Monte Carlo simulations (default: 2000)')
    parser.add_argument('--draw-multiplier', type=float, default=1.0,
                       help='Draw probability multiplier (default: 1.0, higher = more draws)')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving results to file')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip creating visualization')
    parser.add_argument('--tiebreak', choices=['aro_c1', 'rating_only'], default='aro_c1',
                       help='Tiebreak system: aro_c1 (FIDE ARO-C1) or rating_only (default: aro_c1)')
    parser.add_argument('--total-rounds', type=int, default=11,
                       help='Total number of tournament rounds (default: 11)')
    
    args = parser.parse_args()
    
    # Default CSV files
    csv_files = [
        'data/Chess Data - Round 1.csv',
        'data/Chess Data - Round 2.csv',
        'data/Chess Data - Round 3.csv',
        'data/Chess Data - Round 4.csv',
        'data/Chess Data - Round 5.csv',
        'data/Chess Data - Round 6.csv'
    ]
    
    # Initialize predictor
    predictor = ChessTournamentPredictor()
    
    # Run analysis
    analysis = predictor.run_complete_analysis(
        csv_files=csv_files,
        num_simulations=args.simulations,
        save_results=not args.no_save,
        create_plot=not args.no_plot,
        draw_probability_multiplier=args.draw_multiplier,
        tiebreak_system=args.tiebreak,
        total_rounds=args.total_rounds
    )
    
    return analysis

if __name__ == "__main__":
    main()
