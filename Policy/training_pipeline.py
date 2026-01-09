"""
Training pipeline skeleton.
"""

# TODO: Fetch/load data for the target ticker (raw OHLCV + any required metadata).
# TODO: Build the feature matrix from raw data (feature_engineering + scaling).
# TODO: Add labels (pivot/ATR labels, state machine labels, etc.).
# TODO: Split data into train/val/test (time-based split; avoid leakage).
# TODO: Train the model using each label type (iterate label configs + log metrics).
# TODO: Create a new DataFrame that combines model probabilities, state machine
#       outputs, and general market features for RL inputs.
# TODO: Feed the combined features into the RL agent and train the policy.
