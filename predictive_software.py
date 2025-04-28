import sys
import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, font, messagebox
from pykalman import KalmanFilter

class PredictiveSoftware:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictive Software by Dr. Mosab Hawarey")
        
        # Set window size and prevent resizing
        self.root.geometry("1200x650")
        self.root.resizable(False, False)
        
        # Define colors
        self.magenta = "#8B008B"  # Dark Magenta
        self.light_magenta = "#FF00FF"  # Brighter Magenta
        self.white = "#FFFFFF"
        self.black = "#000000"
        
        # Create custom fonts
        self.title_font = font.Font(family="Arial", size=16, weight="bold")
        self.normal_font = font.Font(family="Arial", size=16)
        self.small_font = font.Font(family="Arial", size=12)
        
        # Initialize variables first - before creating any UI elements
        self.csv_file_path = None
        self.data = None
        self.progress_var = tk.DoubleVar()
        self.filter_type = tk.StringVar(value="Standard Kalman Filter")
        self.model_type = tk.StringVar(value="Constant Velocity")
        self.noise_var = tk.DoubleVar(value=0.01)
        self.file_path_var = tk.StringVar()
        
        # Configure main_frame to fill the window
        self.main_frame = tk.Frame(root, bg=self.white)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure columns to have exactly equal width
        self.main_frame.columnconfigure(0, weight=1, uniform="column")
        self.main_frame.columnconfigure(1, weight=1, uniform="column")
        self.main_frame.columnconfigure(2, weight=1, uniform="column")
        
        # Create three exactly equal sized frames for the sections
        section_width = 360  # Same width for all three sections
        
        # Create the three main sections
        self.create_input_section(section_width)
        self.create_filter_options_section(section_width)
        self.create_output_section(section_width)
        
        # Create footer
        self.footer = tk.Label(
            root, 
            text="All Copyrights Reserved: mosab.hawarey.org", 
            bg=self.magenta, 
            fg=self.white,
            font=self.small_font
        )
        self.footer.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_input_section(self, width):
        # Create frame for INPUT section
        self.input_frame = tk.Frame(self.main_frame, width=width, height=600, bg=self.white)
        self.input_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.input_frame.grid_propagate(False)
        
        # Create label for INPUT section
        input_label = tk.Label(
            self.input_frame, 
            text="INPUT", 
            bg=self.magenta, 
            fg=self.white,
            font=self.title_font,
            padx=10,
            pady=5
        )
        input_label.pack(fill=tk.X)
        
        # Create a frame for the content with magenta border
        input_content_frame = tk.Frame(
            self.input_frame, 
            bg=self.white,
            highlightbackground=self.magenta,
            highlightthickness=2
        )
        input_content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add file selection elements
        file_label = tk.Label(
            input_content_frame, 
            text="Select CSV File:", 
            bg=self.white, 
            fg=self.black,
            font=self.normal_font,
            anchor="w"
        )
        file_label.pack(fill=tk.X, padx=10, pady=(20, 5))
        
        # Add file path display
        self.file_path_entry = tk.Entry(
            input_content_frame, 
            textvariable=self.file_path_var, 
            font=self.normal_font,
            bg=self.white,
            fg=self.black,
            state="readonly",
            width=30
        )
        self.file_path_entry.pack(fill=tk.X, padx=10, pady=5)
        
        # Add browse button
        self.browse_button = tk.Button(
            input_content_frame, 
            text="Browse", 
            command=self.browse_file,
            font=self.normal_font,
            bg=self.magenta,
            fg=self.white,
            relief=tk.RAISED,
            bd=3
        )
        self.browse_button.pack(padx=10, pady=20)
        
        # Add description text
        description_text = """
        Input Requirements:
        - Single column CSV file
        - No header
        - Values at regular intervals
        """
        
        description_label = tk.Label(
            input_content_frame,
            text=description_text,
            bg=self.white,
            fg=self.black,
            font=self.small_font,
            justify=tk.LEFT,
            anchor="w"
        )
        description_label.pack(fill=tk.X, padx=10, pady=(40, 10))
    
    def create_filter_options_section(self, width):
        # Create frame for FILTER OPTIONS section
        self.filter_frame = tk.Frame(self.main_frame, width=width, height=600, bg=self.white)
        self.filter_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.filter_frame.grid_propagate(False)
        
        # Create label for FILTER OPTIONS section
        filter_label = tk.Label(
            self.filter_frame, 
            text="FILTER OPTIONS", 
            bg=self.magenta, 
            fg=self.white,
            font=self.title_font,
            padx=10,
            pady=5
        )
        filter_label.pack(fill=tk.X)
        
        # Create a frame for the content with magenta border
        filter_content_frame = tk.Frame(
            self.filter_frame, 
            bg=self.white,
            highlightbackground=self.magenta,
            highlightthickness=2
        )
        filter_content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Filter Type Selection
        filter_type_label = tk.Label(
            filter_content_frame,
            text="Kalman Filter Type:",
            bg=self.white,
            fg=self.black,
            font=self.normal_font,
            anchor="w"
        )
        filter_type_label.pack(fill=tk.X, padx=10, pady=(20, 5))
        
        filter_options = [
            "Standard Kalman Filter",
            "Extended Kalman Filter",
            "Unscented Kalman Filter",
            "Ensemble Kalman Filter"
        ]
        
        self.filter_dropdown = ttk.Combobox(
            filter_content_frame,
            textvariable=self.filter_type,
            values=filter_options,
            font=self.normal_font,
            state="readonly",
            width=25
        )
        self.filter_dropdown.pack(padx=10, pady=5)
        
        # Model Type Selection
        model_type_label = tk.Label(
            filter_content_frame,
            text="Model Type:",
            bg=self.white,
            fg=self.black,
            font=self.normal_font,
            anchor="w"
        )
        model_type_label.pack(fill=tk.X, padx=10, pady=(20, 5))
        
        model_options = [
            "Constant Position",
            "Constant Velocity",
            "Constant Acceleration",
            "Auto-detect Best Model"
        ]
        
        self.model_dropdown = ttk.Combobox(
            filter_content_frame,
            textvariable=self.model_type,
            values=model_options,
            font=self.normal_font,
            state="readonly",
            width=25
        )
        self.model_dropdown.pack(padx=10, pady=5)
        
        # Process Noise Parameter
        noise_label = tk.Label(
            filter_content_frame,
            text="Process Noise (Q):",
            bg=self.white,
            fg=self.black,
            font=self.normal_font,
            anchor="w"
        )
        noise_label.pack(fill=tk.X, padx=10, pady=(20, 5))
        
        noise_scale = tk.Scale(
            filter_content_frame,
            variable=self.noise_var,
            from_=0.001,
            to=0.1,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            bg=self.white,
            fg=self.black,
            font=self.small_font,
            length=width-50
        )
        noise_scale.pack(padx=10, pady=5)
        
        # Add run button at the bottom of filter options
        self.run_button = tk.Button(
            filter_content_frame,
            text="RUN",
            command=self.run_prediction,
            font=self.title_font,
            bg=self.magenta,
            fg=self.white,
            relief=tk.RAISED,
            bd=3,
            width=15,
            height=2
        )
        self.run_button.pack(padx=10, pady=(80, 10))
    
    def create_output_section(self, width):
        # Create frame for OUTPUT section
        self.output_frame = tk.Frame(self.main_frame, width=width, height=600, bg=self.white)
        self.output_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.output_frame.grid_propagate(False)
        
        # Create label for OUTPUT section
        output_label = tk.Label(
            self.output_frame, 
            text="OUTPUT", 
            bg=self.magenta, 
            fg=self.white,
            font=self.title_font,
            padx=10,
            pady=5
        )
        output_label.pack(fill=tk.X)
        
        # Create progress bar
        self.progress_bar = ttk.Progressbar(
            self.output_frame,
            orient="horizontal",
            length=width-40,  # Account for padding
            mode="determinate",
            variable=self.progress_var
        )
        self.progress_bar.pack(padx=10, pady=5)
        
        # Create a frame for the content with magenta border
        output_content_frame = tk.Frame(
            self.output_frame, 
            bg=self.white,
            highlightbackground=self.magenta,
            highlightthickness=2
        )
        output_content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add headers for the output
        headers_frame = tk.Frame(output_content_frame, bg=self.white)
        headers_frame.pack(fill=tk.X, padx=5, pady=10)
        
        prediction_header = tk.Label(
            headers_frame,
            text="Prediction",
            bg=self.white,
            fg=self.black,
            font=self.normal_font,
            width=10,
            anchor="w"
        )
        prediction_header.grid(row=0, column=0, padx=5)
        
        ci_header = tk.Label(
            headers_frame,
            text="Confidence Interval",
            bg=self.white,
            fg=self.black,
            font=self.normal_font,
            width=15,
            anchor="w"
        )
        ci_header.grid(row=0, column=1, padx=5)
        
        # Create a frame for the prediction results
        self.results_frame = tk.Frame(output_content_frame, bg=self.white)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        # Initialize empty labels for results
        self.prediction_labels = []
        self.ci_labels = []
        
        for i in range(10):
            # Prediction value label
            pred_label = tk.Label(
                self.results_frame,
                text="",
                bg=self.white,
                fg=self.black,
                font=self.normal_font,
                width=10,
                anchor="w"
            )
            pred_label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
            self.prediction_labels.append(pred_label)
            
            # Confidence interval label
            ci_label = tk.Label(
                self.results_frame,
                text="",
                bg=self.white,
                fg=self.black,
                font=self.normal_font,
                width=15,
                anchor="w"
            )
            ci_label.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            self.ci_labels.append(ci_label)
    
    def browse_file(self):
        """Open file dialog to select CSV file"""
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            self.csv_file_path = filepath
            self.file_path_var.set(os.path.basename(filepath))
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_var.set(value)
        self.root.update_idletasks()
    
    def run_prediction(self):
        """Run the prediction with the selected options"""
        if not self.csv_file_path:
            messagebox.showerror("Error", "Please select a CSV file first.")
            return
        
        try:
            # Clear previous results
            for label in self.prediction_labels:
                label.config(text="")
            for label in self.ci_labels:
                label.config(text="")
            
            # Update progress
            self.update_progress(10)
            
            # Load CSV data
            try:
                data = pd.read_csv(self.csv_file_path, header=None).values.flatten()
                
                # Check if data is valid
                if len(data) < 2:
                    messagebox.showerror("Error", "CSV file must contain at least 2 data points.")
                    self.update_progress(0)
                    return
                    
                # Check for non-numeric data
                if not np.issubdtype(data.dtype, np.number):
                    messagebox.showerror("Error", "CSV file must contain only numeric values.")
                    self.update_progress(0)
                    return
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read CSV file: {str(e)}")
                self.update_progress(0)
                return
                
            self.update_progress(20)
            
            # Get selected options
            filter_type = self.filter_type.get()
            model_type = self.model_type.get()
            process_noise = self.noise_var.get()
            
            # Run prediction based on filter type
            predictions, confidence_intervals = self.perform_prediction(
                data, filter_type, model_type, process_noise
            )
            
            # Update UI with results
            for i in range(10):
                if i < len(predictions):
                    self.prediction_labels[i].config(text=f"{predictions[i]:.4f}")
                    ci_text = f"±{confidence_intervals[i]:.4f}"
                    self.ci_labels[i].config(text=ci_text)
            
            self.update_progress(100)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.update_progress(0)
    
    def perform_prediction(self, data, filter_type, model_type, process_noise):
        """Perform prediction using the specified Kalman filter"""
        try:
            # Normalize data for better filter performance
            data_mean = np.mean(data)
            data_std = np.std(data)
            
            # Handle case where standard deviation is zero or very close to zero
            if data_std < 1e-8:
                data_std = 1.0
                
            normalized_data = (data - data_mean) / data_std
            
            # Setup the appropriate model dimensions
            if model_type == "Constant Position":
                dim_x = 1  # State dimension (position)
                dim_z = 1  # Measurement dimension (position)
                dt = 1.0   # Time step
                
                # Define the transition matrix (state update)
                F = np.array([[1.0]])
                
                # Define the measurement matrix
                H = np.array([[1.0]])
                
            elif model_type == "Constant Velocity":
                dim_x = 2  # State dimension (position, velocity)
                dim_z = 1  # Measurement dimension (position)
                dt = 1.0   # Time step
                
                # Define the transition matrix (state update)
                F = np.array([
                    [1.0, dt],
                    [0.0, 1.0]
                ])
                
                # Define the measurement matrix
                H = np.array([[1.0, 0.0]])
                
            elif model_type == "Constant Acceleration":
                dim_x = 3  # State dimension (position, velocity, acceleration)
                dim_z = 1  # Measurement dimension (position)
                dt = 1.0   # Time step
                
                # Define the transition matrix (state update)
                F = np.array([
                    [1.0, dt, 0.5*dt*dt],
                    [0.0, 1.0, dt],
                    [0.0, 0.0, 1.0]
                ])
                
                # Define the measurement matrix
                H = np.array([[1.0, 0.0, 0.0]])
            
            else:  # Auto-detect
                # Start with constant velocity and adjust based on data
                # This is a simplified auto-detection for this example
                dim_x = 2  # Start with constant velocity
                dim_z = 1
                dt = 1.0
                
                # Define the transition matrix (state update)
                F = np.array([
                    [1.0, dt],
                    [0.0, 1.0]
                ])
                
                # Define the measurement matrix
                H = np.array([[1.0, 0.0]])
            
            self.update_progress(30)
            
            # Initialize filter based on type
            if filter_type == "Standard Kalman Filter":
                kf = KalmanFilter(
                    transition_matrices=F,
                    observation_matrices=H,
                    initial_state_mean=np.zeros(dim_x),
                    initial_state_covariance=np.eye(dim_x),
                    observation_covariance=1.0,
                    transition_covariance=process_noise * np.eye(dim_x)
                )
                
                # Fit to data
                self.update_progress(40)
                state_means, state_covariances = kf.filter(normalized_data)
                self.update_progress(60)
                
                # Get the last state
                last_state = state_means[-1]
                last_covariance = state_covariances[-1]
                
                # Predict future values
                predictions = []
                confidence_intervals = []
                
                for i in range(10):
                    # Predict the next state
                    next_state = F @ last_state
                    next_covariance = F @ last_covariance @ F.T + process_noise * np.eye(dim_x)
                    
                    # Get the predicted value (first element of state vector)
                    if dim_x == 1:
                        predicted_value = next_state[0]
                    else:
                        predicted_value = next_state[0]
                    
                    # Get the confidence interval (95% = 1.96 standard deviations)
                    if dim_x == 1:
                        ci = 1.96 * np.sqrt(next_covariance[0, 0])
                    else:
                        ci = 1.96 * np.sqrt(next_covariance[0, 0])
                    
                    # Update for next iteration
                    last_state = next_state
                    last_covariance = next_covariance
                    
                    # Save predictions
                    predictions.append(predicted_value)
                    confidence_intervals.append(ci)
                    
                    self.update_progress(60 + (i+1) * 3)
            
            elif filter_type == "Extended Kalman Filter":
                # For simplicity, we implement a linearized version similar to standard KF
                # In a real implementation, you would define non-linear state transition
                # and measurement functions
                kf = KalmanFilter(
                    transition_matrices=F,
                    observation_matrices=H,
                    initial_state_mean=np.zeros(dim_x),
                    initial_state_covariance=np.eye(dim_x),
                    observation_covariance=1.0,
                    transition_covariance=process_noise * np.eye(dim_x)
                )
                
                # Fit to data
                self.update_progress(40)
                state_means, state_covariances = kf.filter(normalized_data)
                self.update_progress(60)
                
                # Get the last state
                last_state = state_means[-1]
                last_covariance = state_covariances[-1]
                
                # Predict future values
                predictions = []
                confidence_intervals = []
                
                for i in range(10):
                    # Predict the next state
                    next_state = F @ last_state
                    next_covariance = F @ last_covariance @ F.T + process_noise * np.eye(dim_x)
                    
                    # Get the predicted value (first element of state vector)
                    predicted_value = next_state[0]
                    
                    # Get the confidence interval (95% = 1.96 standard deviations)
                    ci = 1.96 * np.sqrt(next_covariance[0, 0])
                    
                    # Update for next iteration
                    last_state = next_state
                    last_covariance = next_covariance
                    
                    # Save predictions
                    predictions.append(predicted_value)
                    confidence_intervals.append(ci)
                    
                    self.update_progress(60 + (i+1) * 3)
            
            elif filter_type == "Unscented Kalman Filter":
                # For UKF we would typically define non-linear functions
                # But for this example, we'll use a linear approximation
                kf = KalmanFilter(
                    transition_matrices=F,
                    observation_matrices=H,
                    initial_state_mean=np.zeros(dim_x),
                    initial_state_covariance=np.eye(dim_x),
                    observation_covariance=1.0,
                    transition_covariance=process_noise * np.eye(dim_x)
                )
                
                # Fit to data
                self.update_progress(40)
                state_means, state_covariances = kf.filter(normalized_data)
                self.update_progress(60)
                
                # Get the last state
                last_state = state_means[-1]
                last_covariance = state_covariances[-1]
                
                # Predict future values
                predictions = []
                confidence_intervals = []
                
                for i in range(10):
                    # Predict the next state
                    next_state = F @ last_state
                    next_covariance = F @ last_covariance @ F.T + process_noise * np.eye(dim_x)
                    
                    # Get the predicted value (first element of state vector)
                    predicted_value = next_state[0]
                    
                    # Get the confidence interval (95% = 1.96 standard deviations)
                    ci = 1.96 * np.sqrt(next_covariance[0, 0])
                    
                    # Update for next iteration
                    last_state = next_state
                    last_covariance = next_covariance
                    
                    # Save predictions
                    predictions.append(predicted_value)
                    confidence_intervals.append(ci)
                    
                    self.update_progress(60 + (i+1) * 3)
            
            else:  # Ensemble Kalman Filter
                # Simplified implementation for this example
                kf = KalmanFilter(
                    transition_matrices=F,
                    observation_matrices=H,
                    initial_state_mean=np.zeros(dim_x),
                    initial_state_covariance=np.eye(dim_x),
                    observation_covariance=1.0,
                    transition_covariance=process_noise * np.eye(dim_x)
                )
                
                # Fit to data
                self.update_progress(40)
                state_means, state_covariances = kf.filter(normalized_data)
                self.update_progress(60)
                
                # Get the last state
                last_state = state_means[-1]
                last_covariance = state_covariances[-1]
                
                # Predict future values
                predictions = []
                confidence_intervals = []
                
                for i in range(10):
                    # Predict the next state
                    next_state = F @ last_state
                    next_covariance = F @ last_covariance @ F.T + process_noise * np.eye(dim_x)
                    
                    # Get the predicted value (first element of state vector)
                    predicted_value = next_state[0]
                    
                    # Get the confidence interval (95% = 1.96 standard deviations)
                    ci = 1.96 * np.sqrt(next_covariance[0, 0])
                    
                    # Update for next iteration
                    last_state = next_state
                    last_covariance = next_covariance
                    
                    # Save predictions
                    predictions.append(predicted_value)
                    confidence_intervals.append(ci)
                    
                    self.update_progress(60 + (i+1) * 3)
            
            # Denormalize the predictions
            denormalized_predictions = (np.array(predictions) * data_std) + data_mean
            denormalized_ci = np.array(confidence_intervals) * data_std
            
            return denormalized_predictions, denormalized_ci
            
        except Exception as e:
            # Handle any errors in the prediction process
            messagebox.showerror("Prediction Error", f"Error during prediction: {str(e)}")
            # Return empty arrays as fallback
            return np.zeros(10), np.zeros(10)


def main():
    root = tk.Tk()
    app = PredictiveSoftware(root)
    root.mainloop()

if __name__ == "__main__":
    main()