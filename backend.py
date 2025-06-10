from flask import Flask, jsonify, request
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import io
import base64
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define port from environment variable, default to 5000
port = int(os.getenv('PORT', 5000))

class FIRFilterDesigner:
    def __init__(self):
        self.filter_types = ['Low-Pass', 'High-Pass', 'Band-Pass', 'Band-Stop']
        self.filter_descriptions = {
            'Low-Pass': 'Allows low frequencies to pass through while attenuating high frequencies.',
            'High-Pass': 'Allows high frequencies to pass through while attenuating low frequencies.',
            'Band-Pass': 'Allows a specific frequency band to pass through while attenuating others.',
            'Band-Stop': 'Attenuates a specific frequency band while allowing others to pass.'
        }
        self.parameter_types = ['Cutoff Frequency', 'Transition Bandwidth', 'Passband Ripple', 'Stopband Attenuation']
        self.parameter_descriptions = {
            'Cutoff Frequency': 'The frequency where the filter begins to attenuate the signal.',
            'Transition Bandwidth': 'The frequency range over which the filter transitions.',
            'Passband Ripple': 'The maximum variation in the passband magnitude response.',
            'Stopband Attenuation': 'The minimum attenuation required in the stopband.'
        }

    def get_test_frequencies(self, filter_type, params):
        fs = params[0]
        nyquist = fs / 2
        if filter_type == 'Low-Pass':
            cutoff = params[1]
            width = params[2]
            f_pass = cutoff / 2
            f_stop = min(cutoff + width / 2, nyquist)
        elif filter_type == 'High-Pass':
            cutoff = params[1]
            width = params[2]
            f_pass = min(cutoff + width / 2, nyquist)
            f_stop = max(0, (cutoff - width) / 2)
        elif filter_type == 'Band-Pass':
            lower_freq = params[1]
            upper_freq = params[2]
            width = params[3]
            f_pass = (lower_freq + upper_freq) / 2
            if lower_freq - width / 2 > 0:
                f_stop = lower_freq - width / 2
            elif upper_freq + width / 2 < nyquist:
                f_stop = upper_freq + width / 2
            else:
                f_stop = nyquist / 2  # Fallback
        elif filter_type == 'Band-Stop':
            lower_freq = params[1]
            upper_freq = params[2]
            width = params[3]
            f_stop = (lower_freq + upper_freq) / 2
            if lower_freq - width / 2 > 0:
                f_pass = lower_freq - width / 2
            elif upper_freq + width / 2 < nyquist:
                f_pass = upper_freq + width / 2
            else:
                f_pass = nyquist / 2  # Fallback
        return f_pass, f_stop

    def design_filter(self, filter_type, params):
        fs = params[0]
        nyquist = fs / 2

        # Design the FIR filter
        if filter_type in ['Low-Pass', 'High-Pass']:
            width = params[2] / nyquist
            numtaps = int(4 / width)
            if numtaps % 2 == 0:
                numtaps += 1
            if filter_type == 'Low-Pass':
                b = signal.firwin(numtaps, params[1] / nyquist, pass_zero=True)
            else:  # High-Pass
                b = signal.firwin(numtaps, params[1] / nyquist, pass_zero=False)
        elif filter_type in ['Band-Pass', 'Band-Stop']:
            width = params[3] / nyquist
            numtaps = int(4 / width)
            if filter_type == 'Band-Pass':
                b = signal.remez(numtaps, [0, params[1] - width, params[1], params[2], params[2] + width, nyquist], [0, 1, 0], fs=fs)
            else:  # Band-Stop
                b = signal.remez(numtaps, [0, params[1] - width, params[1], params[2], params[2] + width, nyquist], [1, 0, 1], fs=fs)

        # Frequency response plot
        w, h = signal.freqz(b)
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        axs[0].plot(w / np.pi * nyquist, 20 * np.log10(np.abs(h)), 'b')
        axs[0].set_title(f'Magnitude Response of {filter_type} Filter')
        axs[0].set_xlabel('Frequency (Hz)')
        axs[0].set_ylabel('Gain (dB)')
        axs[0].grid()
        axs[0].axvline(params[1], color='r', linestyle='--')
        if filter_type in ['Band-Pass', 'Band-Stop']:
            axs[0].axvline(params[2], color='r', linestyle='--')
        axs[1].plot(w / np.pi * nyquist, np.angle(h), 'b')
        axs[1].set_title('Phase Response')
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Phase (radians)')
        axs[1].grid()
        plt.tight_layout()

        # Encode frequency response plot
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        freq_response_img = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)

        # Generate test signal
        duration = 1.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        f_pass, f_stop = self.get_test_frequencies(filter_type, params)
        x = np.sin(2 * np.pi * f_pass * t) + np.sin(2 * np.pi * f_stop * t)
        y = signal.lfilter(b, 1, x)  # Apply filter

        # Signal plots
        fig2, axs2 = plt.subplots(2, 1, figsize=(14, 6))
        plot_samples = int(fs * 0.1)  # First 0.1 seconds
        t_plot = t[:plot_samples]
        x_plot = x[:plot_samples]
        y_plot = y[:plot_samples]

        axs2[0].plot(t_plot, x_plot)
        axs2[0].set_title('Original Signal')
        axs2[0].set_xlabel('Time (s)')
        axs2[0].set_ylabel('Amplitude')
        axs2[0].grid()

        axs2[1].plot(t_plot, y_plot)
        axs2[1].set_title('Filtered Signal')
        axs2[1].set_xlabel('Time (s)')
        axs2[1].set_ylabel('Amplitude')
        axs2[1].grid()

        plt.tight_layout()

        # Encode signal plots
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png')
        buf2.seek(0)
        signal_plots_img = base64.b64encode(buf2.read()).decode('utf-8')
        buf2.close()
        plt.close(fig2)

        return freq_response_img, signal_plots_img

@app.route('/api/design_filter', methods=['POST'])
def design_filter():
    data = request.json
    filter_type = data.get('filter_type')
    params = data.get('params')

    designer = FIRFilterDesigner()
    if filter_type in designer.filter_types:
        try:
            freq_response, signal_plots = designer.design_filter(filter_type, params)
            return jsonify({
                "message": "Filter designed successfully",
                "frequency_response": freq_response,
                "signal_plots": signal_plots
            })
        except Exception as e:
            return jsonify({"error": f"Error designing filter: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid filter type"}), 400

@app.route('/')
def home():
    return "Welcome to the FIR Filter Designer API"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False)
