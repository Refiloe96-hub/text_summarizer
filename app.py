from flask import Flask, render_template, request, jsonify, flash, send_from_directory, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
#nltk.download('wordnet')      #download if using this module for the first time

import nltk
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
nltk.download('stopwords')    #download if using this module for the first time

from api_communication import upload, save_transcript

#For Gensim
import gensim
import string
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize

import os
import time
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from datetime import datetime

import magic
import pdfkit

from pydub.utils import mediainfo
from pydub import AudioSegment

import librosa
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploaded_videos')

TIME_LIMIT = 24 * 3600  # 24 hours

from pydub import AudioSegment

ffmpeg_base_path = "C:/Users/22951687/Downloads/Text_Summarizer/ffmpeg-2023-10-18-git-e7a6bba51a-essentials_build/bin"
ffmpeg_path = os.path.join(ffmpeg_base_path, "ffmpeg.exe")
ffprobe_path = os.path.join(ffmpeg_base_path, "ffprobe.exe")

AudioSegment.converter = ffmpeg_path
AudioSegment.ffmpeg = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path


def cleanup_old_files():
    now = time.time()
    
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        if os.path.getmtime(file_path) < now - TIME_LIMIT:
            os.remove(file_path)
            print(f"Deleted {filename}")

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

app = Flask(__name__)

# Database configurations
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///summaries.db'
db = SQLAlchemy(app)

class Summary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    topics = db.Column(db.String(255), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()  # This will initialize the database tables within the application context.

# Set the maximum allowed upload size to 100 MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

from google.cloud import storage
import os

MODEL_DIR = os.path.join(BASE_DIR, 'models_dir')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

credentials_path = os.path.join(BASE_DIR, 'premium-student-402620-299cd6f8fc7d.json')

# Define the function to download from GCS
def download_model_from_gcs(bucket_name, source_blob_prefix, destination_dir):
    try:
        storage_client = storage.Client.from_service_account_json(credentials_path, project="premium-student-402620")
        bucket = storage_client.get_bucket(bucket_name)
        blobs = storage_client.list_blobs(bucket_name, prefix=source_blob_prefix)
        
        for blob in blobs:
            # Creating the full local filepath
            local_file_path = os.path.join(destination_dir, blob.name)
            
            # Creating directories if they do not exist
            local_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            
            # Downloading the file
            blob.download_to_filename(local_file_path)
            print(f"{blob.name} downloaded to {local_file_path}.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Download model files from GCS
bucket_name = 'text_summarizer_model'
model_files_prefix = 'huggingface/hub/models--google--pegasus-cnn_dailymail'
download_model_from_gcs(bucket_name, model_files_prefix, MODEL_DIR)

# Load the Pegasus tokenizer and model
model_name = os.path.join(MODEL_DIR, model_files_prefix)
tokenizer = PegasusTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)


def get_remote_address():
    return request.headers.get('X-Forwarded-For', request.remote_addr)

# Initialize the limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,  # Use the client's IP address to track the rate limit
    default_limits=["100 per day", "20 per hour"]  # Here, 100 requests per day and 20 per hour
)

"""""
# Before sending the file for transcription, you should check if the file contains audio streams
def has_audio(file_path):
    info = mediainfo(file_path)
    if 'audio_streams' in info and int(info['audio_streams']) > 0:
        return True
    return False

def is_audio_clear(file_path, threshold=-50.0):
    audio = AudioSegment.from_file(file_path, format="mp4")  # Assuming the video is in mp4 format
    
    # Check the average amplitude
    avg_amplitude = audio.dBFS

    if avg_amplitude < threshold:
        return False
    return True

#Remember, these techniques are heuristic and might produce false positives or negatives.
#It's essential to choose thresholds wisely and potentially allow users to override the checks 
# if they believe their audio is clear.
def has_sufficient_audio(file_path, silence_threshold=0.5, frame_length=44100):
    
    y, sr = librosa.load(file_path, sr=None)
    non_silent = librosa.effects.split(y, top_db=20, frame_length=frame_length)

    non_silent_duration = sum([end - start for start, end in non_silent]) / sr
    total_duration = len(y) / sr

    if non_silent_duration / total_duration < silence_threshold:
        return False
    return True
 """   

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def text_summarization():
    if request.method == "POST":
        inputtext = request.form["inputtext_"]
        if not inputtext.strip():  # Check for empty input
            return "Input text is empty", 400

        summary, topics = summarize(inputtext)

        # Save the summary to the database
        try:
            summary_record = Summary(text=summary, topics=",".join(topics))
            db.session.add(summary_record)
            db.session.commit()
        except Exception as e:
            # Log the error for debugging
            print(f"Error while saving to database: {e}")
            return "An error occurred while saving the summary.", 500

        return render_template("output.html", data={"summary": summary, "topics": topics})
    
def summarize(inputtext):
    input_text = "summarize: " + inputtext

    tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=1024, truncation=True).to(device)
    summary_ = model.generate(tokenized_text, min_length=30, max_length=300)
    summary = tokenizer.decode(summary_[0], skip_special_tokens=True)

    topics = topic_identifier(summary)

    return summary, topics

ITEMS_PER_PAGE = 10  # Number of summaries displayed per page
from flask import request

ITEMS_PER_PAGE = 10  # Number of summaries displayed per page

@app.route('/history')
def history():
    # Get the current page number from the request arguments (default to 1 if not provided)
    page = request.args.get('page', 1, type=int)
    
    # Fetch a paginated set of summaries from the database, ordered by timestamp in descending order
    summaries = Summary.query.order_by(Summary.timestamp.desc()).paginate(page, ITEMS_PER_PAGE, False)
    print("Current Page:", page)
    print("Number of Summaries Fetched:", len(summaries.items))
    print("Next Page:", summaries.next_num)
    print("Previous Page:", summaries.prev_num)

    # Render the template with the paginated summaries
    return render_template('history.html', summaries=summaries.items, next_url=summaries.next_num, prev_url=summaries.prev_num)

ALLOWED_EXTENSIONS = {'mp4', 'm4v' 'avi', 'mov', 'flv', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

ALLOWED_MIME_TYPES = ['video/mp4', 'video/m4v' 'video/x-msvideo', 'video/quicktime', 'video/x-flv', 'video/x-matroska']

import subprocess

@app.route('/video-upload', methods=['POST'])
@limiter.limit("5 per minute")  # This is just an example, limit to 5 requests per minute
def video_upload():
    file = request.files['file']
    # Check if a file was uploaded
    if not file or file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    # Check if the file type is allowed
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file type"}), 400
    
    # Save the file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    # Compress the video using ffmpeg.exe
    compressed_filepath = os.path.join(UPLOAD_DIR, 'compressed_' + filename)
    subprocess.run([ffmpeg_path, "-i", filepath, "-c:v", "libx264", compressed_filepath])

    audio_url = upload(compressed_filepath)
    if not audio_url:
        return jsonify({"success": False, "error": "Failed to upload video"}), 400

    transcript_text = save_transcript(audio_url, filename)
    if "Error" in transcript_text:
        return transcript_text, 400

    if not transcript_text:
        return jsonify({"success": False, "error": "Unable to generate transcript."}), 400
    
    summary, topics = summarize(transcript_text)
    return jsonify({"success": True, "summary": summary, "topics": topics})

def topic_identifier(text):
    tokens = word_tokenize(text)
    lowercase_tokens = [t.lower() for t in tokens]
    # print(lowercase_tokens)

    #bagofwords approach
    bagofwords_1 = Counter(lowercase_tokens)
    # print(bagofwords_1.most_common(10))

    #Text processing
    alphabets = [t for t in lowercase_tokens if t.isalpha()]

    words = stopwords.words("english")
    stopwords_removed = [t for t in alphabets if t not in words]

    # print(stopwords_removed)

    lemmatizer = WordNetLemmatizer()

    lem_tokens = [lemmatizer.lemmatize(t) for t in stopwords_removed]

    bag_words = Counter(lem_tokens)
    #print(bag_words.most_common(6))
    
    return [word for word, count in bag_words.most_common(6)]

@app.route('/download-txt/<filename>')
def download_txt(filename):
    return send_from_directory(BASE_DIR, filename, as_attachment=True, mimetype='text/plain')


@app.route('/download-pdf/<filename>')
def download_pdf(filename):
    # Fetch the summary based on the filename or another unique identifier
    # Assuming filename is the unique id for the summary
    summary = Summary.query.filter_by(id=filename).first()
    if not summary:
        return "Summary not found", 404

    # Convert the summary text to PDF
    pdf = pdfkit.from_string(summary.text, False)
    
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'inline; filename={filename}.pdf'
    
    return response

from flask import Response
import io

@app.route('/download_csv', methods=['POST'])
def download_csv():
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    
    # Fetch summaries within the specified date range
    summaries = Summary.query.filter(Summary.timestamp.between(start_date, end_date)).all()
    
    # Generate CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(["ID", "Summary", "Topics", "Timestamp"])  # Header
    for summary in summaries:
        writer.writerow([summary.id, summary.text, ", ".join(summary.topics), summary.timestamp])
    
    output.seek(0)
    
    return Response(output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=summaries.csv"})

# Error Handlers
@app.errorhandler(RequestEntityTooLarge)
def handle_file_size_large(e):
    return jsonify({"success": False, "error": "File size too large. Maximum allowed size is 100MB."}), 413


if __name__ == '__main__': # It Allows You to Execute Code When the File Runs as a Script
    app.run(debug=True) 