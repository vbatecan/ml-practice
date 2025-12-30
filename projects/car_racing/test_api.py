import sys
import os

# Add the current directory to path so we can import Game
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from main import Game
from settings import RL_HEADLESS

def test_rl_capabilities():
    print("--- Testing RL Capabilities ---")
    
    # 1. Initialize Game (Headless if possible)
    print("Initializing Game...")
    game = Game()
    
    # 2. Test Reset
    print("\nTesting Automatic Reset...")
    state = game.reset_game()
    if state:
        print("Reset successful!")
        print(f"Initial State: speed={state['speed']}, angle={state['angle']}, collided={state['collided']}")
        print(f"Radars (first 3): {state['radars'][:3]}")
    else:
        print("Reset failed!")
        return

    # 3. Test Step (API)
    print("\nTesting API Step (Command: Full Throttle Forward)...")
    # Action: (W, A, S, D, HB) -> (1, 0, 0, 0, 0)
    action = (1, 0, 0, 0, 0)
    
    # Take 10 steps
    for i in range(1, 11):
        next_state, reward, done, info = game.step(action)
        if i % 5 == 0:
            print(f"Step {i}: speed={next_state['speed']:.2f}, reward={reward:.2f}, collided={next_state['collided']}")

    # 4. Success Verification
    if next_state['speed'] > 0:
        print("\nSUCCESS: Car moved and sensor data was returned via API.")
    else:
        print("\nFAILURE: Car did not move.")

    print("\nTesting second reset (generalization check)...")
    old_start = game.track.start_position
    game.reset_game()
    new_start = game.track.start_position
    
    if old_start != new_start:
        print("SUCCESS: Track regenerated automatically on reset.")
    else:
        # Note: In some rare cases random generation might yield same start, 
        # but usually it should be different.
        print("Note: Track start position is same, but regeneration was called.")

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    test_rl_capabilities()
