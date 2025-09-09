# Chess Tournament Predictor

A pure Monte Carlo Tree Search (MCTS) system for predicting Swiss chess tournament outcomes based on Elo ratings and tournament history.

## Overview

This system simulates chess tournaments by:
1. **Parsing tournament data** from CSV files containing round results
2. **Using Elo ratings** to calculate win/draw/loss probabilities for matchups
3. **Running Monte Carlo simulations** to predict tournament winners
4. **Generating statistical analysis** of win probabilities and tournament dynamics

## Features

- ✅ **Swiss Tournament Simulation**: Uses the `swissdutch` library for accurate pairing
- ✅ **Elo-Based Predictions**: Calculates realistic win/draw/loss probabilities
- ✅ **Monte Carlo Analysis**: Runs thousands of simulations for statistical accuracy
- ✅ **Comprehensive Results**: Win probabilities, expected scores, and inequality metrics
- ✅ **Visualization**: Generates charts and saves results to files

## Installation

```bash
# Install required packages
pip install swissdutch matplotlib pandas requests

# Clone or download the project files
```

## Usage

### Basic Usage

```bash
# Run with default settings (2000 simulations, draw multiplier 1.0)
python chess_tournament_predictor.py

# Run with custom number of simulations
python chess_tournament_predictor.py --simulations 5000

# Run with higher draw probability (more draws = more even tournament)
python chess_tournament_predictor.py --draw-multiplier 1.5

# Run with lower draw probability (fewer draws = more decisive games)
python chess_tournament_predictor.py --draw-multiplier 0.5

# Run without saving files
python chess_tournament_predictor.py --no-save --no-plot

# Run with FIDE tiebreak system (default)
python chess_tournament_predictor.py --tiebreak aro_c1

# Run with rating-only tiebreak (simpler system)
python chess_tournament_predictor.py --tiebreak rating_only

# Run with custom total rounds
python chess_tournament_predictor.py --total-rounds 9
```

### Command Line Arguments

- `--simulations`: Number of Monte Carlo simulations (default: 2000)
- `--draw-multiplier`: Draw probability multiplier (default: 1.0, higher = more draws)
- `--tiebreak`: Tiebreak system - `aro_c1` (FIDE ARO-C1) or `rating_only` (default: aro_c1)
- `--total-rounds`: Total number of tournament rounds (default: 11)
- `--no-save`: Skip saving results to file
- `--no-plot`: Skip creating visualization

### Programmatic Usage

```python
from chess_tournament_predictor import ChessTournamentPredictor

# Initialize predictor
predictor = ChessTournamentPredictor()

# Load tournament data
csv_files = [
    'Chess Data - Round 1.csv',
    'Chess Data - Round 2.csv', 
    'Chess Data - Round 3.csv',
    'Chess Data - Round 4.csv'
]

# Run complete analysis
analysis = predictor.run_complete_analysis(
    csv_files=csv_files,
    num_simulations=2000,
    save_results=True,
    create_plot=True,
    draw_probability_multiplier=1.0
)

# Access results
win_probabilities = analysis['win_probabilities']
top_3_probability = analysis['top_3_probability']
```

## Input Data Format

The system expects CSV files with the following structure:
- **SNo**: Player serial number
- **NAME**: Player name
- **Rtg**: Elo rating
- **Title**: FIDE title (GM, IM, FM, etc.)
- **Round results**: Columns showing wins (1), losses (0), draws (0.5)

Example:
```csv
SNo,NAME,Rtg,Title,Rd.1,Rd.2,Rd.3,Rd.4
1,Player A,2700,GM,1,0.5,1,0
2,Player B,2650,GM,0,1,0.5,1
```

## Output

### Console Output
- Top 10 win probabilities
- Summary statistics (top 3 probability, total simulations, etc.)
- Winner inequality analysis (Gini coefficient)

### Files Generated
- `tournament_prediction_YYYYMMDD_HHMMSS.json`: Complete results data
- `tournament_prediction_YYYYMMDD_HHMMSS.png`: Visualization chart

### Example Output
```
🏆 CHESS TOURNAMENT PREDICTION RESULTS
================================================================================

📊 TOP 10 WIN PROBABILITIES:
Rank Player                         Win %    Expected Score
------------------------------------------------------------
1    Praggnanandhaa, R              18.8%    1.6
2    Erigaisi, Arjun                14.2%    1.2
3    Gukesh, D                      11.2%    0.9
4    Keymer, Vincent                6.7%     0.6
5    Abdusattorov, Nodirbek         6.2%     0.5

📈 SUMMARY STATISTICS:
   Top 3 Combined Probability: 44.2%
   Total Simulations: 2,000
   Unique Winners: 45
   Winner Inequality (Gini): 0.623
   📊 Moderate inequality - some clear favorites
```

## System Components

### Core Files
- `chess_tournament_predictor.py`: Main prediction system
- `chess_tournament_parser.py`: CSV data parsing and player management
- `monte_carlo_tournament.py`: Monte Carlo simulation engine
- `elo_predictor.py`: Elo-based probability calculations

### Key Classes
- `ChessTournamentPredictor`: Main orchestrator class
- `ChessTournamentParser`: Handles CSV parsing and player state management
- `MonteCarloTournament`: Runs tournament simulations
- `EloPredictor`: Calculates win/draw/loss probabilities

## Algorithm Details

### 1. Data Loading
- Parses CSV files with duplicate column handling
- Merges player data across multiple rounds
- Calculates current scores, opponents, and color history

### 2. Elo Probability Calculation
- Uses standard Elo formula: `E = 1 / (1 + 10^((Rb - Ra) / 400))`
- Converts expected score to win/draw/loss probabilities
- Applies Kirill Kryukov's draw probability model: `-R.Diff/32.49 + exp((Av.Rating - 2254.7)/208.49) + 23.87`

### 3. Tournament Simulation
- Seeds with completed rounds (Rounds 1-N based on available data)
- Simulates remaining rounds (Rounds N+1 to total_rounds) using Swiss pairing
- Uses FIDE ARO-C1 tiebreak system by default (configurable)
- Uses probabilistic outcome determination based on Elo

### 4. Monte Carlo Analysis
- Runs thousands of independent simulations
- Collects winner frequencies and statistics
- Calculates win probabilities and confidence intervals

## Configuration

### Simulation Parameters
- `num_simulations`: Number of Monte Carlo runs (default: 2000)
- `draw_probability_multiplier`: Adjusts draw likelihood (default: 1.0)
  - `1.0`: Normal draw probability
  - `>1.0`: Higher draw probability (more even tournament)
  - `<1.0`: Lower draw probability (more decisive games)

### Elo Parameters
- `K_factor`: Elo rating change factor (default: 32)
- `draw_probability_formula`: Kirill Kryukov's model: `-R.Diff/32.49 + exp((Av.Rating - 2254.7)/208.49) + 23.87`
  - `Av.Rating`: Average rating of both players
  - `R.Diff`: Absolute difference in ratings
  - Source: https://kirill-kryukov.com/chess/kcec/draw_rate.html

## Tiebreak Systems

The system supports two tiebreak methods for player ranking:

### FIDE ARO-C1 (Default)
- **Primary**: Score (points)
- **Secondary**: Average Rating of Opponents Cut 1 (ARO-C1)
- **Tertiary**: Player rating
- **Advantage**: Rewards players who faced stronger opponents
- **Realism**: Matches official FIDE Swiss tournament regulations

### Rating-Only
- **Primary**: Score (points)  
- **Secondary**: Player rating
- **Advantage**: Simpler system, purely rating-based
- **Use case**: When you want to ignore strength of schedule

## Draw Probability Multiplier Effects

The `--draw-multiplier` parameter significantly affects tournament outcomes:

### Higher Draw Probability (multiplier > 1.0)
- **Effect**: More games end in draws
- **Result**: Tournament becomes more even, more players have similar scores
- **Example**: `--draw-multiplier 1.5` makes the tournament more competitive

### Lower Draw Probability (multiplier < 1.0)  
- **Effect**: Fewer games end in draws, more decisive results
- **Result**: Tournament becomes more top-heavy, stronger players dominate
- **Example**: `--draw-multiplier 0.5` makes the tournament more decisive

### Typical Values
- `0.5`: Very decisive games (few draws)
- `1.0`: Normal draw rate (default)
- `1.5`: More draws, more even tournament
- `2.0`: High draw rate, very competitive

## Performance

- **Simulation Speed**: ~100-200 simulations per second
- **Memory Usage**: Minimal (only stores final results)
- **Accuracy**: Improves with more simulations (diminishing returns after 2000)

## Troubleshooting

### Common Issues
1. **CSV parsing errors**: Check file format and column names
2. **Missing players**: Ensure all players appear in at least one round
3. **Low simulation count**: Increase `--simulations` for better accuracy

### File Requirements
- CSV files must be in the same directory
- Files should be named consistently (e.g., "Chess Data - Round X.csv")
- All rounds must have the same player set

## License

This project is for educational and research purposes. Please respect tournament data usage policies.

## Contributing

Feel free to submit issues or improvements. The system is designed to be modular and extensible.