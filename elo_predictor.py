import math
from typing import NamedTuple, Tuple

class MatchupProbabilities(NamedTuple):
    """Container for win, draw, and loss probabilities"""
    win_prob: float
    draw_prob: float
    loss_prob: float
    
    def __str__(self):
        return f"Win: {self.win_prob:.1%}, Draw: {self.draw_prob:.1%}, Loss: {self.loss_prob:.1%}"

class EloPredictor:
    """
    Predicts win, draw, and loss probabilities for chess matchups based on Elo ratings.
    
    Uses the standard Elo formula for expected scores and estimates draw probability
    based on the average rating of the players.
    """
    
    def __init__(self, draw_probability_multiplier: float = 1.0):
        """
        Initialize the Elo predictor.
        
        Args:
            draw_probability_multiplier: Multiplier to adjust draw probability (default: 1.0)
        """
        self.draw_probability_multiplier = draw_probability_multiplier
    
    def calculate_expected_score(self, player_rating: int, opponent_rating: int) -> float:
        """
        Calculate the expected score for a player against an opponent.
        
        Args:
            player_rating: Elo rating of the player
            opponent_rating: Elo rating of the opponent
            
        Returns:
            Expected score (0.0 to 1.0) representing average points per game
        """
        rating_diff = opponent_rating - player_rating
        expected_score = 1 / (1 + 10 ** (rating_diff / 400))
        return expected_score
    
    def estimate_draw_probability(self, player_rating: int, opponent_rating: int) -> float:
        """
        Estimate the probability of a draw using the formula from Kirill Kryukov's analysis.
        
        Uses the formula: Draw rate = -R.Diff/32.49 + exp((Av.Rating - 2254.7)/208.49) + 23.87
        Source: https://kirill-kryukov.com/chess/kcec/draw_rate.html
        
        Args:
            player_rating: Elo rating of player A
            opponent_rating: Elo rating of player B
            
        Returns:
            Estimated draw probability (0.0 to 1.0)
        """
        import math
        
        # Calculate average rating and absolute rating difference
        average_rating = (player_rating + opponent_rating) / 2
        rating_diff = abs(player_rating - opponent_rating)
        
        # Apply the Kirill Kryukov formula
        # Draw rate = -R.Diff/32.49 + exp((Av.Rating - 2254.7)/208.49) + 23.87
        draw_rate = -rating_diff / 32.49 + math.exp((average_rating - 2254.7) / 208.49) + 23.87
        
        # Convert percentage to probability (0-1)
        draw_prob = draw_rate / 100.0
        
        # Apply the draw probability multiplier
        draw_prob = draw_prob * self.draw_probability_multiplier
        
        # Ensure draw probability is between 0 and 1
        draw_prob = max(0.0, min(1.0, draw_prob))
        
        return draw_prob
    
    def calculate_matchup_probabilities(self, player_a_rating: int, player_b_rating: int) -> Tuple[MatchupProbabilities, MatchupProbabilities]:
        """
        Calculate win, draw, and loss probabilities for both players in a matchup.
        
        Args:
            player_a_rating: Elo rating of player A
            player_b_rating: Elo rating of player B
            
        Returns:
            Tuple of (PlayerA_probabilities, PlayerB_probabilities)
        """
        # Step 1: Calculate expected scores
        expected_score_a = self.calculate_expected_score(player_a_rating, player_b_rating)
        expected_score_b = self.calculate_expected_score(player_b_rating, player_a_rating)
        
        # Step 2: Estimate draw probability
        draw_prob = self.estimate_draw_probability(player_a_rating, player_b_rating)
        
        # Step 3: Calculate win probabilities
        # E_A = P_Win_A + (0.5 * P_Draw)
        # Therefore: P_Win_A = E_A - (0.5 * P_Draw)
        win_prob_a = expected_score_a - (0.5 * draw_prob)
        win_prob_b = expected_score_b - (0.5 * draw_prob)
        
        # Ensure win probabilities are non-negative and don't exceed reasonable bounds
        win_prob_a = max(0.0, min(1.0 - draw_prob, win_prob_a))
        win_prob_b = max(0.0, min(1.0 - draw_prob, win_prob_b))
        
        # Normalize probabilities to ensure they sum to 1.0
        total_a = win_prob_a + draw_prob + win_prob_b
        if total_a > 0:
            win_prob_a = win_prob_a / total_a
            draw_prob = draw_prob / total_a
            win_prob_b = win_prob_b / total_a
        
        # Step 4: Calculate loss probabilities
        loss_prob_a = win_prob_b
        loss_prob_b = win_prob_a
        
        # Create probability objects
        probs_a = MatchupProbabilities(
            win_prob=win_prob_a,
            draw_prob=draw_prob,
            loss_prob=loss_prob_a
        )
        
        probs_b = MatchupProbabilities(
            win_prob=win_prob_b,
            draw_prob=draw_prob,
            loss_prob=loss_prob_b
        )
        
        return probs_a, probs_b
    
    def predict_matchup(self, player_a_name: str, player_a_rating: int, 
                       player_b_name: str, player_b_rating: int) -> None:
        """
        Predict and display the outcome probabilities for a matchup.
        
        Args:
            player_a_name: Name of player A
            player_a_rating: Elo rating of player A
            player_b_name: Name of player B
            player_b_rating: Elo rating of player B
        """
        probs_a, probs_b = self.calculate_matchup_probabilities(player_a_rating, player_b_rating)
        
        print(f"Matchup: {player_a_name} ({player_a_rating}) vs {player_b_name} ({player_b_rating})")
        print(f"Rating difference: {abs(player_a_rating - player_b_rating)} points")
        print()
        print(f"{player_a_name}: {probs_a}")
        print(f"{player_b_name}: {probs_b}")
        print()
        
        # Verify probabilities sum to 1.0
        total_a = probs_a.win_prob + probs_a.draw_prob + probs_a.loss_prob
        total_b = probs_b.win_prob + probs_b.draw_prob + probs_b.loss_prob
        print(f"Probability sums: {player_a_name} = {total_a:.3f}, {player_b_name} = {total_b:.3f}")
        print("-" * 60)

def main():
    """Example usage and testing"""
    predictor = EloPredictor()
    
    print("=== Elo-Based Chess Matchup Predictions ===\n")
    
    # Test with the example from the description
    print("Example from description:")
    predictor.predict_matchup("Player A", 1800, "Player B", 1600)
    
    # Test with some tournament players
    print("Tournament player examples:")
    predictor.predict_matchup("Praggnanandhaa", 2785, "Gukesh", 2767)
    predictor.predict_matchup("Firouzja", 2754, "Erigaisi", 2771)
    predictor.predict_matchup("Maghsoodloo", 2692, "Niemann", 2733)
    
    # Test with larger rating differences
    print("Large rating differences:")
    predictor.predict_matchup("GM Player", 2700, "IM Player", 2400)
    predictor.predict_matchup("Strong Player", 2500, "Weaker Player", 2000)
    
    # Test with very close ratings
    print("Very close ratings:")
    predictor.predict_matchup("Player X", 2600, "Player Y", 2601)

if __name__ == "__main__":
    main()