import requests
import cv2
import io

class TelegramAlert:
    def __init__(self, bot_token=None, chat_id=None):
        """
        Inicializa el cliente de Telegram.
        Si no se proporcionan token/chat_id, el sistema avisará por consola pero no fallará.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        
        if not self.bot_token or not self.chat_id:
            print("[ADVERTENCIA] Telegram Token o Chat ID no configurados. Las alertas solo se verán en consola.")

    def send_alert(self, message, frame=None):
        """
        Envía un mensaje de texto y opcionalmente una foto del frame capturado.
        """
        print(f"\n[ALERTA LOCAL] {message}\n")
        
        if not self.bot_token or not self.chat_id:
            return False
            
        try:
            if frame is not None:
                # Codificar el frame a JPEG en memoria
                _, buffer = cv2.imencode('.jpg', frame)
                photo_bytes = io.BytesIO(buffer)
                
                url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
                data = {'chat_id': self.chat_id, 'caption': message}
                files = {'photo': ('alert.jpg', photo_bytes, 'image/jpeg')}
                
                response = requests.post(url, data=data, files=files)
            else:
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                data = {'chat_id': self.chat_id, 'text': message}
                response = requests.post(url, data=data)
                
            if response.status_code == 200:
                print("[TELEGRAM] Alerta enviada con éxito.")
                return True
            else:
                print(f"[TELEGRAM ERROR] No se pudo enviar: {response.text}")
                return False
                
        except Exception as e:
            print(f"[TELEGRAM EXCEPCIÓN] {e}")
            return False

if __name__ == "__main__":
    # Prueba
    alert = TelegramAlert(bot_token="AQUI_TU_TOKEN", chat_id="AQUI_TU_CHAT_ID")
    alert.send_alert("⚠️ Alerta de prueba desde el sistema de seguridad.")
