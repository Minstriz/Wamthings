from flask import Flask, render_template, Response, request, redirect, url_for, flash, session, jsonify, send_file
from flask_socketio import SocketIO, emit, disconnect
from functools import wraps
import os
import cv2
import numpy as np
import datetime
import random
import time
import pandas as pd
from threading import Thread, Lock
from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox
import csv
import logging
import asyncio
import telegram
import json
from report_excel import generate_excel_report

# Cấu hình Telegram
TELEGRAM_TOKEN = '7941539579:AAHKZeWa4rfp_Zk06hgCzjSk5yp_CcnZWgQ'
TELEGRAM_CHAT_ID = '6262392731'

# Đường dẫn file JSON
JSON_FILE = 'attendance_data.json'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'abc123412123'
app.config['SESSION_TYPE'] = 'filesystem'
socketio = SocketIO(app)

# Paths
LOG_FILE = 'attendance_log.csv'
FACES_DIR = './faces'
DET_MODEL = './weights/det_10g.onnx'
REC_MODEL = './weights/w600k_r50.onnx'

# Locks
video_lock = Lock()
log_lock = Lock()

# Check files
required_files = {
    "Detection model": DET_MODEL,
    "Recognition model": REC_MODEL,
}

for name, path in required_files.items():
    if not os.path.exists(path):
        logger.error(f"{name} not found at {path}")
        raise FileNotFoundError(f"{name} not found at {path}")

if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)
    logger.info(f"Created directory: {FACES_DIR}")

# Init log file
def init_log_file():
    try:
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Thời gian", "Tên", "Check-in", "Check-out"])
            logger.info(f"Created new log file: {LOG_FILE}")
        else:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if not lines or lines[0].strip() != "Thời gian,Tên,Check-in,Check-out":
                with open(LOG_FILE, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Thời gian", "Tên", "Check-in", "Check-out"])
                    for line in lines[1:]:
                        cols = line.strip().split(',')
                        if len(cols) >= 4:
                            writer.writerow(cols[:4])
                logger.info(f"Fixed log file structure: {LOG_FILE}")
    except Exception as e:
        logger.error(f"Error initializing log file: {e}")
        raise

init_log_file()

# Init JSON file
def init_json_file():
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        logger.info(f"Created new JSON file: {JSON_FILE}")

init_json_file()

# Load models
try:
    detector = SCRFD(DET_MODEL, input_size=(640, 640), conf_thres=0.5)
    recognizer = ArcFace(REC_MODEL)
    logger.info("Face detection and recognition models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

def build_targets():
    targets = []
    for filename in os.listdir(FACES_DIR):
        try:
            name = os.path.splitext(filename)[0]
            path = os.path.join(FACES_DIR, filename)
            image = cv2.imread(path)
            if image is None:
                logger.warning(f"Could not read image: {path}")
                continue
                
            bboxes, kpss = detector.detect(image, max_num=1)
            if len(kpss) == 0:
                logger.warning(f"No face detected in: {path}")
                continue
                
            embedding = recognizer(image, kpss[0])
            targets.append((embedding, name))
            logger.info(f"Added face target: {name}")
        except Exception as e:
            logger.error(f"Error processing target {filename}: {e}")
    
    logger.info(f"Built {len(targets)} face targets in total")
    return targets

targets = build_targets()
colors = {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
          for _, name in targets}

last_log_time = 0
COOLDOWN_SECONDS = 5

async def send_telegram_message(bot, message):
    """
    Gửi tin nhắn qua Telegram.
    """
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        logger.info(f"Gửi tin nhắn Telegram thành công: {message}")
    except Exception as e:
        logger.error(f"Lỗi khi gửi tin nhắn Telegram: {str(e)}")

def log_to_json(data):
    """
    Ghi dữ liệu vào file JSON và gửi thông báo Telegram.
    """
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        headers = ["Thời gian", "Tên", "Check-in", "Check-out"]
        for row in data:
            record = {headers[i]: str(row[i]) for i in range(len(row))}
            json_data.append(record)
            
            if record['Check-in'] or record['Check-out']:
                bot = telegram.Bot(token=TELEGRAM_TOKEN)
                message = (
                    f"New record:\n"
                    f"Time: {record['Thời gian']}\n"
                    f"Name: {record['Tên']}\n"
                    f"{'Check-in' if record['Check-in'] else 'Check-out'}: "
                    f"{record['Check-in'] or record['Check-out']}"
                )
                asyncio.run(send_telegram_message(bot, message))
        
        with open(JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Đã ghi dữ liệu vào {JSON_FILE}")
    except Exception as e:
        logger.error(f"Lỗi khi ghi file JSON hoặc gửi Telegram: {str(e)}")
        bot = telegram.Bot(token=TELEGRAM_TOKEN)
        asyncio.run(send_telegram_message(bot, f"Lỗi khi ghi dữ liệu JSON: {str(e)}"))

def log_attendance(name, status):
    global last_log_time
    current_time = time.time()
    
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    check_in = time_now if status == "checkin" else ""
    check_out = time_now if status == "checkout" else ""
    
    if name and name != "Unknown" and current_time - last_log_time >= COOLDOWN_SECONDS:
        try:
            with log_lock:
                safe_name = name.replace(',', '_')
                with open(LOG_FILE, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([time_now, safe_name, check_in, check_out])
                
                json_data = [[time_now, safe_name, check_in, check_out]]
                log_to_json(json_data)
                
                try:
                    df = pd.read_csv(LOG_FILE, on_bad_lines='skip').tail(5)[::-1]
                    socketio.emit('attendance_update', df.to_dict(orient='records'), namespace='/admin')
                except Exception as e:
                    logger.error(f"Error reading CSV for socketio: {e}")
            
            last_log_time = current_time
            logger.info(f"Logged attendance: {name}, {status}")
            return True, f"Đã ghi {status} thành công cho {name}"
        except Exception as e:
            logger.error(f"Error logging attendance: {e}")
            return False, f"Lỗi khi ghi {status}: {str(e)}"
    else:
        logger.info(f"Attendance not logged: {name}, {status}, cooldown: {current_time - last_log_time}")
        return False, "Không ghi được: Tên không hợp lệ hoặc trong thời gian chờ"

def create_error_frame(message):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    if len(message) > 40:
        words = message.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(' '.join(current_line + [word])) <= 40:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                
        if current_line:
            lines.append(' '.join(current_line))
            
        y_pos = 220
        for line in lines:
            cv2.putText(frame, line, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_pos += 40
    else:
        cv2.putText(frame, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

def generate_frames(action, user_agent):
    for attempt in range(2):
        if video_lock.acquire(blocking=True, timeout=0.5):
            cap = None
            try:
                logger.info("Starting generate_frames")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    logger.error("Could not open camera")
                    error_frame = create_error_frame("Không thể mở camera")
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n'), (None, "Không thể mở camera")
                    return

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                
                detector_input_size = (640, 640)
                detector.input_size = detector_input_size
                
                name_detected = None
                success = False
                message = ""
                face_detected = False
                start_time = time.time()
                duration = 5
                frame_count = 0
                recognition_complete = False
                no_face_alert_shown = False
                
                while time.time() - start_time < duration:
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Could not read frame")
                        error_frame = create_error_frame("Không thể đọc frame")
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n'), (None, "Không thể đọc frame")
                        time.sleep(0.1)
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    frame_count += 1
                    
                    if not recognition_complete and frame_count % 2 == 0:
                        try:
                            h, w = frame.shape[:2]
                            scale = min(detector_input_size[0] / w, detector_input_size[1] / h)
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            
                            frame_resized = cv2.resize(frame, (new_w, new_h))
                            frame_for_detection = np.zeros((detector_input_size[1], detector_input_size[0], 3), dtype=np.uint8)
                            
                            x_offset = (detector_input_size[0] - new_w) // 2
                            y_offset = (detector_input_size[1] - new_h) // 2
                            frame_for_detection[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_resized
                            
                            bboxes, kpss = detector.detect(frame_for_detection, max_num=1)
                            
                            if len(bboxes) == 0 and time.time() - start_time > duration / 2 and not no_face_alert_shown:
                                no_face_alert_shown = True
                                
                            if len(bboxes) > 0:
                                face_detected = True
                                no_face_alert_shown = False
                                
                                for i, (bbox, kps) in enumerate(zip(bboxes, kpss)):
                                    bbox_original = bbox.copy()
                                    bbox_original[0] = (bbox_original[0] - x_offset) / scale
                                    bbox_original[1] = (bbox_original[1] - y_offset) / scale
                                    bbox_original[2] = (bbox_original[2] - x_offset) / scale
                                    bbox_original[3] = (bbox_original[3] - y_offset) / scale
                                    
                                    kps_original = kps.copy()
                                    for j in range(kps.shape[0]):
                                        kps_original[j][0] = (kps_original[j][0] - x_offset) / scale
                                        kps_original[j][1] = (kps_original[j][1] - y_offset) / scale
                                    
                                    embedding = recognizer(frame, kps_original)
                                    best_match = "Unknown"
                                    max_sim = 0
                                    
                                    for emb_target, name in targets:
                                        sim = compute_similarity(embedding, emb_target)
                                        if sim > max_sim and sim > 0.6:
                                            max_sim = sim
                                            best_match = name
                                            
                                    if best_match != "Unknown":
                                        color = colors.get(best_match, (0, 255, 0))
                                        draw_bbox(frame, bbox_original.astype(int), color)
                                        name_detected = best_match
                                    else:
                                        draw_bbox(frame, bbox_original.astype(int), (0, 0, 255))
                                        name_detected = "Unknown"
                                        
                                    recognition_complete = True
                                    
                                    if name_detected and name_detected != "Unknown":
                                        success, message = log_attendance(name_detected, action)
                                        
                        except Exception as e:
                            logger.error(f"Error in face detection/recognition: {e}")
                            error_frame = create_error_frame(f"Lỗi nhận diện: {str(e)}")
                            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n'), (None, f"Lỗi nhận diện: {str(e)}")
                            time.sleep(0.1)
                            continue
                    
                    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    frame_bytes = buffer.tobytes()
                    logger.info(f"Sending frame, size: {len(frame_bytes)} bytes")
                    
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'), (success, name_detected, message)
                    
                    time.sleep(0.03)
                
                empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', empty_frame)
                empty_frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + empty_frame_bytes + b'\r\n'), (success, name_detected, message)
                
                if not face_detected and not no_face_alert_shown:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + empty_frame_bytes + b'\r\n'), (False, "Không phát hiện khuôn mặt", "Không phát hiện khuôn mặt")
                elif name_detected == "Unknown" and recognition_complete:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + empty_frame_bytes + b'\r\n'), (False, "Không nhận diện được khuôn mặt", "Không nhận diện được khuôn mặt")
                    
            except Exception as e:
                logger.error(f"Error in generate_frames: {e}")
                error_frame = create_error_frame(f"Lỗi camera: {str(e)}")
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n'), (None, f"Lỗi camera: {str(e)}", f"Lỗi camera: {str(e)}")
            finally:
                if cap and cap.isOpened():
                    cap.release()
                if video_lock.locked():
                    video_lock.release()
                logger.info("Camera released")
            return
        logger.warning(f"Video lock not acquired, attempt {attempt + 1}/3")
        time.sleep(0.5)
    
    error_frame = create_error_frame("Video đang được sử dụng")
    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n'), (None, "Video đang được sử dụng", "Video đang được sử dụng")

# Hard-coded users
users = {
    "user1": {"password": "pass123", "role": "user"},
    "admin1": {"password": "admin123", "role": "admin"}
}

def login_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'username' not in session or session['role'] != role:
                flash("Vui lòng đăng nhập!", "error")
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/checkin', methods=['POST'])
@login_required('user')
def checkin():
    action = 'checkin'
    user_agent = request.headers.get('User-Agent', '').lower()
    
    for _, result in generate_frames(action, user_agent):
        success, name_detected, message = result
        if success:
            flash(message, "success")
        elif message:
            flash(message, "error")
        break
    return redirect(url_for('index'))

@app.route('/checkout', methods=['POST'])
@login_required('user')
def checkout():
    action = 'checkout'
    user_agent = request.headers.get('User-Agent', '').lower()
    
    for _, result in generate_frames(action, user_agent):
        success, name_detected, message = result
        if success:
            flash(message, "success")
        elif message:
            flash(message, "error")
        break
    return redirect(url_for('index'))

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']

        if username in users and users[username]["password"] == password and users[username]["role"] == role:
            session['username'] = username
            session['role'] = role
            if role == 'user':
                return redirect(url_for('index'))
            elif role == 'admin':
                return redirect(url_for('admin'))
        else:
            flash("Sai tên đăng nhập hoặc mật khẩu!", "error")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    flash("Đã đăng xuất!", "success")
    return redirect(url_for('login'))

@app.route('/index')
@login_required('user')
def index():
    return render_template('index.html')

@app.route('/admin', methods=['GET', 'POST'])
@login_required('admin')
def admin():
    global targets, colors

    if request.method == 'POST':
        if 'file' not in request.files or 'name' not in request.form:
            flash("Vui lòng chọn file và nhập tên!", "error")
        else:
            file = request.files['file']
            name = request.form['name'].strip()
            if file.filename == '' or not name:
                flash("File hoặc tên không hợp lệ!", "error")
            elif not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                flash("Chỉ hỗ trợ file .jpg, .jpeg, .png!", "error")
            else:
                filename = f"{name}.jpg"
                filepath = os.path.join(FACES_DIR, filename)
                if os.path.exists(filepath):
                    flash("Tên nhân viên đã tồn tại!", "error")
                else:
                    try:
                        file.save(filepath)
                        image = cv2.imread(filepath)
                        if image is None:
                            os.remove(filepath)
                            flash("Không thể đọc file ảnh!", "error")
                        else:
                            bboxes, kpss = detector.detect(image, max_num=1)
                            if len(kpss) == 0:
                                os.remove(filepath)
                                flash("Không phát hiện khuôn mặt trong ảnh!", "error")
                            else:
                                targets = build_targets()
                                colors = {n: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                                          for _, n in targets}
                                flash(f"Đã thêm nhân viên {name} thành công!", "success")
                    except Exception as e:
                        logger.error(f"Error adding employee: {e}")
                        flash(f"Lỗi khi thêm nhân viên: {str(e)}", "error")

    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines='skip').tail(5)[::-1]
        data = df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error reading admin CSV: {e}")
        flash("Không tìm thấy file log!", "error")
        data = []
    return render_template('admin.html', data=data)

@app.route('/get_attendance')
@login_required('admin')
def get_attendance():
    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines='skip').tail(5)[::-1]
        data = df.to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error reading attendance CSV: {e}")
        return jsonify([])

@app.route('/generate-report', methods=['GET'])
@login_required('admin')
def generate_report():
    try:
        excel_file = generate_excel_report()
        if excel_file and os.path.exists(excel_file):
            return send_file(
                excel_file,
                as_attachment=True,
                download_name='attendance_report.xlsx',
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            logger.error("Failed to generate Excel report")
            return jsonify({"error": "Không thể tạo file Excel"}), 500
    except Exception as e:
        logger.error(f"Error in generate_report: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/video_feed')
@login_required('user')
def video_feed():
    action = request.args.get('action', 'checkin')
    user_agent = request.headers.get('User-Agent', '').lower()
    
    def stream():
        for frame, _ in generate_frames(action, user_agent):
            yield frame
    
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect', namespace='/admin')
def handle_connect():
    if 'username' not in session or session['role'] != 'admin':
        disconnect()
    else:
        logger.info('Admin connected to /admin namespace')

if __name__ == '__main__':
    socketio.run(app, debug=True)