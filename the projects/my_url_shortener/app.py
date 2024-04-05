from flask import Flask, render_template, request, redirect, url_for
import random
import string

app = Flask(__name__)

# Dictionary to store short URLs and their corresponding long URLs
url_mapping = {}

# Generate a random short code for the URL
def generate_short_code():
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(6))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/shorten', methods=['POST'])
def shorten():
    long_url = request.form.get('long_url')
    if long_url:
        short_code = generate_short_code()
        url_mapping[short_code] = long_url
        return render_template('shortened.html', short_url=f"http://localhost:5000/{short_code}")
    else:
        return "Invalid URL. Please enter a valid URL."

@app.route('/<short_code>')
def redirect_to_long_url(short_code):
    if short_code in url_mapping:
        long_url = url_mapping[short_code]
        return redirect(long_url)
    else:
        return "Short URL not found."

if __name__ == '__main__':
    app.run(debug=True)
