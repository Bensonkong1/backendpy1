from flask import Flask, jsonify, request
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import io
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Define your local backend server port
port = 5000

class FIRFilterDesigner:
    def __init__(self):
        self.filter_types = ['Low-Pass', 'High-Pass', 'Band-Pass', 'Band-Stop']
        self.filter_descriptions = {
            'Low-Pass': 'Allows low frequencies to pass through while attenuating high frequencies.',
            'High-Pass': 'Allows high frequencies to pass through while attenuating low frequencies.',
            'Band-Pass': 'Allows a specific frequency band to pass through while attenuating all other frequencies.',
            'Band-Stop': 'Attenuates a specific frequency band while allowing all other frequencies to pass through.'
        }
        self.parameter_types = ['Cutoff Frequency', 'Transition Bandwidth', 'Passband Ripple', 'Stopband Attenuation']
        self.parameter_descriptions = {
            'Cutoff Frequency': 'The frequency at which the filter begins to attenuate the signal.',
            'Transition Bandwidth': 'The range of frequencies over which the filter transitions from passband to stopband.',
            'Passband Ripple': 'The maximum amount of variation in the filter\'s magnitude response within the passband.',
            'Stopband Attenuation': 'The minimum amount of attenuation required in the stopband.'
        }

    def design_filter(self, filter_type, params):
        fs = params[0]
        nyquist = fs / 2

        if filter_type in ['Low-Pass', 'High-Pass']:
            width = params[2] / nyquist  # Normalized transition bandwidth
            numtaps = int(4 / width)  # Approximate number of taps
            if numtaps % 2 == 0:
                numtaps += 1  # Ensure numtaps is odd
            if filter_type == 'Low-Pass':
                b = signal.firwin(numtaps, params[1] / nyquist, pass_zero=True)
            elif filter_type == 'High-Pass':
                b = signal.firwin(numtaps, params[1] / nyquist, pass_zero=False)
        elif filter_type in ['Band-Pass', 'Band-Stop']:
            width = params[3] / nyquist  # Normalized transition bandwidth
            numtaps = int(4 / width)  # Approximate number of taps
            if filter_type == 'Band-Pass':
                b = signal.remez(numtaps, [0, params[1] - width, params[1], params[2], params[2] + width, nyquist], [0, 1, 0], fs=fs)
            elif filter_type == 'Band-Stop':
                b = signal.remez(numtaps, [0, params[1] - width, params[1], params[2], params[2] + width, nyquist], [1, 0, 1], fs=fs)

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

        # Save plot to a BytesIO object and encode to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        return img_str

@app.route('/api/design_filter', methods=['POST'])
def design_filter():
    data = request.json
    print("Received data:", data)  # Debugging line
    filter_type = data.get('filter_type')
    params = data.get('params')

    designer = FIRFilterDesigner()
    if filter_type in designer.filter_types:
        try:
            img_str = designer.design_filter(filter_type, params)
            return jsonify({"message": "Filter designed successfully", "image": img_str})
        except Exception as e:
            print("Error during filter design:", e)  # Debugging line
            return jsonify({"error": "Error designing filter"}), 500
    else:
        return jsonify({"error": "Invalid filter type"}), 400
        
@app.route('/')
def home():
    return "Welcome to the FIR Filter Designer API"

# Run the Flask app
if __name__ == '__main__':
    app.run(port=5000, debug=True)
