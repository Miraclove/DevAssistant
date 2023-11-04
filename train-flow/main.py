from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
import subprocess
from datetime import datetime
import queue
import threading
import time
import os
import shlex


app = Flask(__name__)
socketio = SocketIO(app)

# Global variables
script_queue = queue.Queue()
scripts_in_queue = []
current_script = None
current_output = []
start_time = None
finished_scripts = []
failed_scripts = []

def worker():
    global current_script, current_output, start_time, finished_scripts, failed_scripts, scripts_in_queue
    while True:
        if not script_queue.empty():
            script = script_queue.get()
            scripts_in_queue.remove(script)  # Remove the script from the queue list
            socketio.emit('queue_updated', {'scripts_in_queue': scripts_in_queue})  # Emit queue update
            current_script = script
            socketio.emit('current_script', {'current_script': current_script})  # Emit current script update
            
            current_output = []
            start_time = datetime.now()

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            proc = subprocess.Popen(shlex.split(script), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True, env=env)
            
            for line in iter(proc.stdout.readline, ''):
                current_output.append(line)
                if len(current_output) > 10:  # Limit to the last ten lines
                    current_output.pop(0)
                socketio.emit('script_output', {'data': line})

            retcode = proc.poll()
            

            # update statue
            socketio.emit('script_output', {'data': f"Process finished with exit code {retcode}"})
            socketio.emit('current_script', {'current_script': 'No script is currently running.'})  # Emit current script update
            if retcode == 0:
                finished_scripts.append(script)
                socketio.emit('script_finished', {'script': script})
            else:
                failed_scripts.append(script)
                socketio.emit('script_failed', {'script': script})

            current_script = None
            start_time = None
        else:
            time.sleep(1)


@app.route('/')
def index():
    return render_template('index.html', current_script=current_script, scripts_in_queue=scripts_in_queue, finished_scripts=finished_scripts, failed_scripts=failed_scripts)

@app.route('/add_script', methods=['POST'])
def add_script():
    script = request.form.get('script_path')
    script_queue.put(script)
    scripts_in_queue.append(script)
    return redirect(url_for('index'))


# Start the worker thread
threading.Thread(target=worker, daemon=True).start()
if __name__ == '__main__':
    socketio.run(app, debug=True)
