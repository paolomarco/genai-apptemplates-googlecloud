from flask import Flask, render_template, request, jsonify
import vertexai
from vertexai.preview.language_models import ChatModel
import os
import google.cloud.logging

app = Flask(__name__)
PROJECT_ID = os.environ.get('GCP_PROJECT') #Your Google Cloud Project ID
LOCATION = os.environ.get('GCP_REGION')   #Your Google Cloud Project Region

client = google.cloud.logging.Client(project=PROJECT_ID)
client.setup_logging()

LOG_NAME = "flask-app-internal-logs"
logger = client.logger(LOG_NAME)

vertexai.init(project=PROJECT_ID, location=LOCATION)

def create_session():
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    chat = chat_model.start_chat(
    context="""You are an assistant to write social posts. You will be provided with a topic. Your job is to generate 2 sample articles and ask the user to choose the one they prefer. The articles should be generated and returned to the user in an HTML format so they can be easily integrated.""",
    examples=[
        InputOutputTextPair(
            input_text="""Iphone 15""",
            output_text="""<h1>The iPhone 15 is here!</h1>
<p>The iPhone 15 is the latest and greatest smartphone from Apple, and it's packed with features that will make your life easier.</p>
<ul>
<li>A new A16 Bionic chip that's up to 50 percent faster than the previous generation</li>
<li>A new camera system with a 48MP main sensor and a new ultrawide sensor</li>
<li>A new display with a higher refresh rate and a smaller notch</li>
<li>A new battery that lasts up to 24 hours on a single charge</li>
</ul>
<p>The iPhone 15 is available now in four colors: black, white, blue, and red.</p>
<p>Order yours today!</p>
<h1>The iPhone 15: Should you upgrade?</h1>
<p>The iPhone 15 is the latest and greatest smartphone from Apple, but is it worth upgrading from your current iPhone?</p>
<ul>
<li>The iPhone 15 has a new A16 Bionic chip that's up to 50 percent faster than the previous generation.</li>
<li>The iPhone 15 has a new camera system with a 48MP main sensor and a new ultrawide sensor.</li>
<li>The iPhone 15 has a new display with a higher refresh rate and a smaller notch.</li>
<li>The iPhone 15 has a new battery that lasts up to 24 hours on a single charge.</li>
</ul>
<p>So, should you upgrade to the iPhone 15? If you're looking for the latest and greatest smartphone, then yes, the iPhone 15 is a great option. However, if you're happy with your current iPhone, then you may not need to upgrade.</p>"""
        )
    ]
)
    return chat

def response(chat, message):
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    result = chat.send_message(message, **parameters)
    return result.text

@app.route('/')
def index():
    ###
    return render_template('index.html')

@app.route('/palm2', methods=['GET', 'POST'])
def vertex_palm():
    user_input = ""
    if request.method == 'GET':
        user_input = request.args.get('user_input')
    else:
        user_input = request.form['user_input']
    logger.log(f"Starting chat session...")
    chat_model = create_session()
    logger.log(f"Chat Session created")
    content = response(chat_model,user_input)
    return jsonify(content=content)

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
