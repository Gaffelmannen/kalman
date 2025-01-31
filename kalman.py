#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, initial_state, initial_uncertainty, process_variance, measurement_variance):
        """
        Initialize the Kalman Filter
        :param initial_state: Initial estimate of state
        :param initial_uncertainty: Initial estimate of uncertainty
        :param process_variance: Variance in the process (how much we trust our model)
        :param measurement_variance: Variance in the measurement (sensor noise)
        """
        self.state = initial_state  # x
        self.uncertainty = initial_uncertainty  # P
        self.process_variance = process_variance  # Q
        self.measurement_variance = measurement_variance  # R

    def predict(self, control_input=0):
        """
        Predict the next state and uncertainty
        """
        # Predict next state (assuming no control input)
        self.state = self.state + control_input
        
        # Increase uncertainty due to process noise
        self.uncertainty = self.uncertainty + self.process_variance

    def update(self, measurement):
        """
        Update the estimate using the measurement
        :param measurement: New measurement
        """
        # Compute Kalman Gain
        kalman_gain = self.uncertainty / (self.uncertainty + self.measurement_variance)

        # Update state estimate
        self.state = self.state + kalman_gain * (measurement - self.state)

        # Update uncertainty
        self.uncertainty = (1 - kalman_gain) * self.uncertainty

    def get_state(self):
        """
        Get the current state estimate
        """
        return self.state

if __name__ == "__main__":
    kf = KalmanFilter(initial_state=0, initial_uncertainty=1, process_variance=0.1, measurement_variance=0.5)

    true_values = np.linspace(0, 10, 500)  # Simulated ground truth
    measurements = true_values + np.random.normal(0, 1.5, size=len(true_values))  

    print("Initial State:", kf.get_state())

    estimated_states = []

    for i, measurement in enumerate(measurements):
        kf.predict()
        kf.update(measurement)
        print(f"Measurement {i+1}: {measurement}, Estimated State: {kf.get_state()}")

    # Plotting results
    plt.figure(figsize=(10, 5))
    plt.plot(true_values, label="True State", linestyle="dashed", color="green")
    plt.scatter(range(len(measurements)), measurements, label="Noisy Measurements", color="red", alpha=0.5)
    plt.plot(estimated_states, label="Kalman Filter Estimate", color="blue")
    
    plt.xlabel("Time Step")
    plt.ylabel("State Value")
    plt.title("Kalman Filter State Estimation")
    plt.legend()
    plt.grid(True)
    plt.show()
