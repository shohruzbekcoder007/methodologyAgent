# API'dan foydalanish

`methodologyagent` servisi manzili: `http://172.16.8.38:9095`

---

## 1. API qisqacha

| Metod | Yo'l | Token kerakmi | Vazifasi |
|---|---|---|---|
| GET | `/health` | yo'q | servis ishlayotganini tekshiradi |
| GET | `/ready` | yo'q | agent tayyorligini tekshiradi (tayyor bo'lmasa `503`) |
| GET | `/v1/info` | ha | model, provayder va vositalar ro'yxati |
| POST | `/v1/chat` | ha | savol-javob |
| GET | `/docs` | yo'q | Swagger UI: http://172.16.8.38:9095/docs |

### `/v1/chat` uchun majburiy sarlavhalar

| Sarlavha | Qiymat | Izoh |
|---|---|---|
| `Authorization` | `Bearer <GATEWAY_TOKEN>` | tokensiz so'rovga `401` qaytadi |
| `X-User-Id` | foydalanuvchi ID'si, masalan `user-42` | har bir foydalanuvchining xotirasi alohida saqlanadi. Ruxsat etilgan belgilar: `A-Z a-z 0-9 . _ -`, ko'pi bilan 64 ta. Sarlavha bo'lmasa `400` qaytadi |
| `Content-Type` | `application/json` | |

### So'rov tanasi

```json
{
  "message": "Inflyatsiya qanday hisoblanadi?",
  "session_id": "suhbat-001",
  "reset_session": false
}
```

- `message`: majburiy, ko'pi bilan 8000 belgi (`MAX_MESSAGE_CHARS`).
- `session_id`: ixtiyoriy. Bir xil qiymat yuborilsa, agent oldingi xabarlarni
  eslab qoladi.
- `reset_session`: `true` bo'lsa, shu sessiya tarixi tozalanadi.

### Javob

```json
{
  "success": true,
  "response": "Inflyatsiya iste'mol narxlari indeksi (INI) orqali ...",
  "session_id": "suhbat-001",
  "tools_called": [{"name": "search_knowledge", "...": "..."}],
  "tool_call_count": 2,
  "backend": "hermes"
}
```

Xatolik bo'lsa `success: false` qaytadi. Bunda `error`, `error_code` va
`retryable` maydonlari to'ldiriladi.

---

## 2. Misollar

Quyidagi misollarda `<GATEWAY_TOKEN>` o'rniga administrator bergan tokenni qo'ying.

### 2.1. curl (Linux / macOS / Git Bash)

```bash
SERVER=http://172.16.8.38:9095
TOKEN=<GATEWAY_TOKEN>

curl -s "$SERVER/v1/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-User-Id: user-42" \
  -H "Content-Type: application/json" \
  -d '{"message":"Aholi soni qanday hisoblanadi?","session_id":"suhbat-001"}'
```

Shu sessiyada davom ettirish:

```bash
curl -s "$SERVER/v1/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-User-Id: user-42" \
  -H "Content-Type: application/json" \
  -d '{"message":"Buni qaysi hujjat tartibga soladi?","session_id":"suhbat-001"}'
```

### 2.2. PowerShell (Windows)

```powershell
$Server = "http://172.16.8.38:9095"
$Headers = @{
  "Authorization" = "Bearer <GATEWAY_TOKEN>"
  "X-User-Id"     = "user-42"
}
$Body = @{
  message    = "YaIM qanday hisoblanadi?"
  session_id = "suhbat-001"
} | ConvertTo-Json

$r = Invoke-RestMethod -Uri "$Server/v1/chat" -Method Post `
  -Headers $Headers -Body ([Text.Encoding]::UTF8.GetBytes($Body)) `
  -ContentType "application/json; charset=utf-8"
$r.response
```

### 2.3. Python

```python
import uuid
import requests

SERVER = "http://172.16.8.38:9095"
TOKEN = "<GATEWAY_TOKEN>"


class MethodologyClient:
    def __init__(self, user_id: str):
        self.s = requests.Session()
        self.s.headers.update({
            "Authorization": f"Bearer {TOKEN}",
            "X-User-Id": user_id,
        })
        self.session_id = str(uuid.uuid4())

    def ask(self, message: str) -> str:
        r = self.s.post(
            f"{SERVER}/v1/chat",
            json={"message": message, "session_id": self.session_id},
            timeout=180,  # LLM va vositalar chaqiruvi vaqt olishi mumkin
        )
        r.raise_for_status()
        data = r.json()
        if not data["success"]:
            raise RuntimeError(f'{data.get("error_code")}: {data.get("error")}')
        return data["response"]


bot = MethodologyClient("user-42")
print(bot.ask("Ishsizlik darajasi qanday hisoblanadi?"))
print(bot.ask("Qaysi manbalardan foydalaniladi?"))  # oldingi savolni eslaydi
```

### 2.4. JavaScript / Node.js (18+)

```js
const SERVER = "http://172.16.8.38:9095";
const TOKEN = "<GATEWAY_TOKEN>";

async function ask(userId, message, sessionId) {
  const res = await fetch(`${SERVER}/v1/chat`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${TOKEN}`,
      "X-User-Id": userId,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message, session_id: sessionId }),
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  const data = await res.json();
  if (!data.success) throw new Error(data.error);
  return data.response;
}

console.log(await ask("user-42", "Tashqi savdo statistikasi metodologiyasi?", "suhbat-001"));
```

> Tokenni brauzerdagi JavaScript kodiga joylashtirmang, chunki uni har kim
> ko'ra oladi. Brauzer ilovasi so'rovni o'z backend'i (gateway) orqali yuborishi
> kerak. Backend foydalanuvchini autentifikatsiya qiladi, keyin token va
> `X-User-Id` ni qo'shib yuboradi.

### 2.5. Sessiyani tozalash

```bash
curl -s "$SERVER/v1/chat" \
  -H "Authorization: Bearer $TOKEN" -H "X-User-Id: user-42" \
  -H "Content-Type: application/json" \
  -d '{"message":"Yangi mavzu","session_id":"suhbat-001","reset_session":true}'
```

---

## 3. Xatolik kodlari

| Kod | Sababi | Nima qilish kerak |
|---|---|---|
| `400` | `X-User-Id` yuborilmagan yoki noto'g'ri formatda | sarlavhani qo'shing, faqat ruxsat etilgan belgilardan foydalaning |
| `401` | token yuborilmagan yoki noto'g'ri | `Authorization: Bearer <GATEWAY_TOKEN>` ni tekshiring |
| `422` | JSON tanasi noto'g'ri (`message` yo'q yoki juda uzun) | tanani tekshiring |
| `429` | so'rovlar chegarasi oshdi (standart: 30/min, burst 10) | `Retry-After` sarlavhasida ko'rsatilgan soniya kuting |
| `503` (`/ready`) | agent hali tayyor emas | birozdan keyin qayta urinib ko'ring |
