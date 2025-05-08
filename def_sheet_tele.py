import json
import os
import logging
from datetime import datetime
import asyncio
import aiohttp
import telegram
import time

# Cấu hình logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler('attendance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cấu hình
JSON_FILE = os.environ.get('JSON_FILE', 'attendance_data.json')
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '7941539579:AAHKZeWa4rfp_Zk06hgCzjSk5yp_CcnZWgQ')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '6262392731')
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Tạo đối tượng bot Telegram
bot = None

async def check_telegram_api():
    """
    Kiểm tra xem API Telegram có hoạt động không bằng cách gọi getUpdates
    """
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{TELEGRAM_API_URL}/getUpdates"
            logger.debug(f"Checking Telegram API: {url}")
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('ok'):
                        logger.info("Telegram API is responsive")
                        return True
                    else:
                        logger.error(f"Telegram API error: {data.get('description', 'Unknown error')}")
                        return False
                else:
                    logger.error(f"Telegram API responded with status code: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"Error checking Telegram API: {str(e)}")
        return False

async def init_telegram_bot():
    """
    Khởi tạo bot Telegram sau khi kiểm tra API
    """
    global bot
    if await check_telegram_api():
        try:
            bot = telegram.Bot(token=TELEGRAM_TOKEN)
            bot_info = await bot.get_me()
            logger.info(f"Bot initialized: @{bot_info.username}")
            return True
        except telegram.error.TelegramError as e:
            logger.error(f"Failed to initialize bot: {str(e)}")
            return False
    return False

async def send_telegram_message(message):
    """
    Gửi tin nhắn qua bot Telegram
    """
    global bot
    try:
        if bot is None:
            if not await init_telegram_bot():
                return await send_telegram_message_direct(message)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        logger.info(f"Telegram message sent: {message}")
        return True
    except telegram.error.TelegramError as e:
        logger.error(f"Telegram API error: {str(e)}")
        return await send_telegram_message_direct(message)
    except Exception as e:
        logger.error(f"Error sending Telegram message: {str(e)}")
        return False

async def send_telegram_message_direct(message):
    """
    Gửi tin nhắn trực tiếp qua HTTP request tới API Telegram sử dụng aiohttp
    """
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{TELEGRAM_API_URL}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={message}"
            logger.debug(f"Sending direct API call: {url}")
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('ok'):
                        logger.info("Direct API message sent")
                        return True
                    else:
                        logger.error(f"Direct API error: {data.get('description', 'Unknown error')}")
                        return False
                else:
                    logger.error(f"Direct API status code: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"Direct API call error: {str(e)}")
        return False

def send_telegram_message_sync(message):
    """
    Gửi tin nhắn Telegram từ mã đồng bộ
    """
    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            success = loop.run_until_complete(send_telegram_message(message))
            if success:
                return True
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay}s (Attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
        except Exception as e:
            logger.error(f"Error in send_telegram_message_sync (attempt {attempt+1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
    logger.error(f"Failed to send message after {max_retries} attempts")
    return False

async def test_telegram_connection_async():
    """
    Kiểm tra kết nối Telegram
    """
    logger.info("Testing Telegram connection...")
    test_message = f"🔄 Test message ({datetime.now().strftime('%H:%M:%S')})"
    success = await send_telegram_message(test_message)
    logger.info(f"Telegram connection {'successful' if success else 'failed'}")
    return success

def test_telegram_connection():
    """
    Phiên bản đồng bộ của hàm kiểm tra kết nối
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(test_telegram_connection_async())
    except Exception as e:
        logger.error(f"Error testing Telegram connection: {str(e)}")
        return False

def init_json_file():
    """Khởi tạo file JSON nếu nó không tồn tại."""
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        logger.info(f"Created new JSON file: {JSON_FILE}")

init_json_file()

def log_to_json(data):
    """
    Ghi dữ liệu vào file JSON và gửi thông báo Telegram.
    """
    logger.debug(f"Received data: {data}")
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        headers = ["Thời gian", "Tên", "Check-in", "Check-out"]
        for row in data:
            formatted_row = [str(value) for value in row]
            while len(formatted_row) < 4:
                formatted_row.append("")
            
            record = {headers[i]: formatted_row[i] for i in range(len(headers))}
            json_data.append(record)
            
            logger.debug(f"Adding record: {record}")
            action_type = "Check-in" if record["Check-in"] else "Check-out"
            action_time = record["Check-in"] if record["Check-in"] else record["Check-out"]
            message = (
                f"New record added:\n"
                f"Time: {record['Thời gian']}\n"
                f"Name: {record['Tên']}\n"
                f"{action_type}: {action_time}"
            )
            logger.debug(f"Attempting to send Telegram message: {message}")
            success = send_telegram_message_sync(message)
            if not success:
                logger.warning("Failed to send Telegram notification")
        
        with open(JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Successfully wrote data to {JSON_FILE}")
        return True
    
    except Exception as e:
        logger.error(f"Error writing to JSON: {str(e)}")
        send_telegram_message_sync(f"Error writing to JSON: {str(e)}")
        return False
