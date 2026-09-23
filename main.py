import asyncio
import logging
import os
import re
import io
from io import BytesIO
from datetime import datetime
import asyncpg
from aiogram import Bot, Dispatcher, types, BaseMiddleware, F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, InlineKeyboardMarkup, InlineKeyboardButton
from openai import AsyncOpenAI
from aiohttp import web

# --- Конфиг ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

ALLOWED_IDS = [5264513480, 8834374199, 5389046699, 2083728480, 6612130539]

db_pool = None
MAX_AUTO_PARTS = 15
CODE_EXTENSIONS = ["html", "py", "js", "css", "java", "cpp", "sql", "json"]


# === RETRY-ОБЁРТКА ДЛЯ GROQ ===
async def call_groq_with_retry(messages, max_retries=3, max_tokens=1200, temperature=0.5):
    """Вызов Groq с retry при 429 (Too Many Requests)."""
    for attempt in range(max_retries):
        try:
            return await client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < max_retries - 1:
                wait = 15 * (attempt + 1)  # 15, 30, 45 сек
                logging.warning(f"[429] Жду {wait} сек (попытка {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                raise


# --- Утилиты ---
def detect_extension(text: str) -> str:
    if "<!DOCTYPE html>" in text or "<html" in text.lower(): return "html"
    if "def " in text and "import " in text: return "py"
    if "function " in text or "const " in text or "let " in text: return "js"
    if "#include" in text: return "cpp"
    if "SELECT " in text and "FROM " in text: return "sql"
    if text.strip().startswith("{") and text.strip().endswith("}"): return "json"
    return "txt"

def clean_code(text: str) -> str:
    text = text.replace("[file content end]", "").replace("[file content begin]", "")
    text = text.replace("// (продолжение следует)", "").replace("// (код готов)", "")
    text = text.replace("(продолжение следует)", "").replace("(код готов)", "")
    text = re.sub(r'^```[\w]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```$', '', text)
    text = text.replace("```html", "").replace("```javascript", "").replace("```js", "")
    text = text.replace("```python", "").replace("```py", "").replace("```css", "")
    text = text.replace("```", "")
    return text.strip()

def extract_code(answer: str) -> str:
    if "```" in answer:
        matches = re.findall(r'```(?:\w+)?\n(.*?)```', answer, re.DOTALL)
        if matches:
            return clean_code("\n".join(m.strip() for m in matches))
    markers = ["<!DOCTYPE", "<html", "def ", "import ", "function ", "const ", "class ", "public class"]
    lines = answer.split("\n")
    start_idx = -1
    for i, line in enumerate(lines):
        if any(m in line for m in markers):
            start_idx = i; break
    if start_idx >= 0:
        return clean_code("\n".join(lines[start_idx:]))
    return clean_code(answer)

def finalize_code(code: str, ext: str) -> str:
    if ext == "html":
        low = code.lower()
        if "</script>" not in low and "<script" in low:
            code += "\n</script>"
        if "</body>" not in low and "<body" in low:
            code += "\n</body>"
        if "</html>" not in low:
            code += "\n</html>"
    return code

def is_code_complete(answer: str) -> bool:
    low = answer.lower()
    if "код готов" in low or "// (готово)" in low: return True
    if "продолжение следует" in low or "to be continued" in low: return False
    if "</html>" in low and "</script>" in low: return True
    if answer.count("```") >= 2 and "</html>" in answer: return True
    return False

def merge_code_parts(parts: list[str], ext: str) -> str:
    if not parts: return ""
    if ext != "html":
        return "\n\n".join(parts)
    result = parts[0]
    for part in parts[1:]:
        part = re.sub(r'^<!DOCTYPE[^>]*>\s*', '', part)
        part = re.sub(r'^<html[^>]*>\s*', '', part)
        part = re.sub(r'^<head>.*?</head>\s*', '', part, flags=re.DOTALL)
        part = re.sub(r'^<body[^>]*>\s*', '', part)
        if "<script" in result and "</script>" not in result.split("<script")[-1]:
            part = re.sub(r'^<script[^>]*>\s*', '', part)
        result += "\n" + part
    return result


# --- БД ---
async def init_db():
    async with db_pool.acquire() as c:
        await c.execute("""CREATE TABLE IF NOT EXISTS code_parts (
            id SERIAL PRIMARY KEY, user_id BIGINT, project_id TEXT, part_num INT,
            content TEXT, topic TEXT, original_request TEXT, file_ext TEXT,
            status TEXT DEFAULT 'in_progress', created_at TIMESTAMP DEFAULT NOW())""")
        await c.execute("ALTER TABLE code_parts ADD COLUMN IF NOT EXISTS original_request TEXT")
        await c.execute("ALTER TABLE code_parts ADD COLUMN IF NOT EXISTS file_ext TEXT")

async def save_code_part(uid: int, pid: str, part: int, content: str, topic: str, original_request: str = "", file_ext: str = ""):
    async with db_pool.acquire() as c:
        await c.execute("""INSERT INTO code_parts (user_id, project_id, part_num, content, topic, original_request, file_ext)
            VALUES ($1, $2, $3, $4, $5, $6, $7)""",
            uid, pid, part, content, topic, original_request, file_ext)

async def get_code_parts(uid: int, pid: str):
    async with db_pool.acquire() as c:
        rows = await c.fetch("SELECT part_num, content FROM code_parts WHERE user_id = $1 AND project_id = $2 ORDER BY part_num", uid, pid)
    return [r['content'] for r in rows]

async def get_code_meta(uid: int, pid: str):
    async with db_pool.acquire() as c:
        r = await c.fetchrow("""SELECT topic, original_request, file_ext FROM code_parts
            WHERE user_id = $1 AND project_id = $2 LIMIT 1""", uid, pid)
    return dict(r) if r else {}

async def get_active_code(uid: int):
    async with db_pool.acquire() as c:
        r = await c.fetchrow("SELECT project_id, topic FROM code_parts WHERE user_id = $1 AND status = 'in_progress' ORDER BY id DESC LIMIT 1", uid)
    return dict(r) if r else None

async def finish_code(uid: int, pid: str):
    async with db_pool.acquire() as c:
        await c.execute("UPDATE code_parts SET status = 'done' WHERE user_id = $1 AND project_id = $2", uid, pid)

async def delete_code(uid: int, pid: str):
    async with db_pool.acquire() as c:
        await c.execute("DELETE FROM code_parts WHERE user_id = $1 AND project_id = $2", uid, pid)


# --- Whisper ---
async def transcribe_audio(file_id: str, ext: str = "ogg") -> str:
    f = await bot.get_file(file_id)
    d = await bot.download_file(f.file_path)
    buf = BytesIO(d.read()); buf.name = f"audio.{ext}"
    t = await client.audio.transcriptions.create(
        model="whisper-large-v3", file=buf, language="ru"
    )
    return t.text


# --- Middleware ---
class AccessMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        if isinstance(event, types.Message):
            if event.from_user.id not in ALLOWED_IDS: return
        return await handler(event, data)

dp.message.middleware(AccessMiddleware())


# --- Промпт ---
CODE_PROMPT = (
    "Ты — код-ассистент. Пишешь ТОЛЬКО чистый код, БЕЗ текста и пояснений. "
    "Пиши ЧАСТЯМИ по 800 токенов. НЕ разрывай строки кода на середине — "
    "заканчивай текущую строку перед обрывом. "
    "В конце ОБЯЗАТЕЛЬНО маркер: `// (продолжение следует)` или `// (код готов)`. "
    "Если просят дописать — читай, что есть, продолжай с последней строки. "
    "НЕ начинай заново. НЕ повторяй функции. "
    "Если это HTML — пиши один файл от <!DOCTYPE> до </html>."
)


# --- Автопромпт ---
async def improve_prompt(user_request: str) -> str:
    try:
        r = await call_groq_with_retry(
            messages=[{"role": "user", "content":
                f"Сделай краткий промпт для генерации кода по запросу: {user_request}\n"
                f"Максимум 3 предложения. Укажи язык, что писать частями по 800 токенов, "
                f"маркеры `// (продолжение следует)` / `// (код готов)`. "
                f"Верни ТОЛЬКО промпт."}],
            max_tokens=150, temperature=0.3
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"[improve_prompt] {e}")
        return user_request


# --- Кнопки ---
def code_keyboard(project_id: str, is_complete: bool = False, auto_mode: bool = False):
    if is_complete:
        return InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Начать заново", callback_data=f"code_restart_{project_id}")]
        ])
    buttons = [
        [
            InlineKeyboardButton(text="📄 Продолжить", callback_data=f"code_cont_{project_id}"),
            InlineKeyboardButton(text="✅ Завершить", callback_data=f"code_done_{project_id}")
        ]
    ]
    if not auto_mode:
        buttons.append([InlineKeyboardButton(text="⏩ Авто (дописать до конца)", callback_data=f"code_auto_{project_id}")])
    buttons.append([InlineKeyboardButton(text="🔄 Начать заново", callback_data=f"code_restart_{project_id}")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


# --- Генерация кода ---
async def make_code(msg: types.Message, request: str):
    uid = msg.from_user.id
    active = await get_active_code(uid)
    if active and "продолж" in request.lower():
        project_id = active['project_id']
    else:
        project_id = f"code_{uid}_{int(datetime.now().timestamp())}"

    status = await msg.answer("💻 Готовлю промпт...")
    try:
        old_parts = await get_code_parts(uid, project_id)
        old_code = "\n\n".join(old_parts) if old_parts else ""
        next_part = len(old_parts) + 1

        if not old_code:
            improved = await improve_prompt(request)
            original_request = request
        else:
            improved = None
            meta = await get_code_meta(uid, project_id)
            original_request = meta.get('original_request', request)

        if old_code:
            prompt = (
                f"ОРИГИНАЛЬНЫЙ ЗАПРОС: {original_request}\n\n"
                f"Продолжи код. Уже написано (последние строки):\n\n"
                f"```\n{old_code[-2500:]}\n```\n\n"
                f"ПИШИ ТОЛЬКО КОД. НЕ начинай заново. НЕ повторяй. "
                f"НЕ разрывай строки на середине. "
                f"Часть {next_part}. Максимум 800 токенов. "
                f"В конце маркер: `// (продолжение следует)` или `// (код готов)`."
            )
        else:
            prompt = (
                f"{improved}\n\n"
                f"ПИШИ ТОЛЬКО КОД. НЕ разрывай строки на середине. "
                f"Максимум 800 токенов. "
                f"В конце маркер: `// (продолжение следует)` или `// (код готов)`."
            )

        r = await call_groq_with_retry(
            messages=[
                {"role": "system", "content": CODE_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200, temperature=0.5
        )
        answer = r.choices[0].message.content
        code = extract_code(answer)
        if not code or len(code.strip()) < 10:
            await status.edit_text("❌ Пустой ответ. Попробуй ещё раз.")
            return

        prev_parts = await get_code_parts(uid, project_id)
        prev_full = "\n\n".join(prev_parts) if prev_parts else ""
        ext = detect_extension(prev_full + "\n" + code)
        if ext == "txt":
            meta = await get_code_meta(uid, project_id)
            ext = meta.get('file_ext', '') or "html"

        await save_code_part(uid, project_id, next_part, code, request[:100], original_request, ext)
        all_parts = await get_code_parts(uid, project_id)
        merged = merge_code_parts(all_parts, ext)
        is_complete = is_code_complete(answer)

        if is_complete:
            merged = finalize_code(merged, ext)
            await finish_code(uid, project_id)
            await msg.answer_document(
                BufferedInputFile(merged.encode("utf-8"), filename=f"final.{ext}"),
                caption=f"✅ Готово. Частей: {len(all_parts)}",
                reply_markup=code_keyboard(project_id, is_complete=True)
            )
        else:
            await msg.answer_document(
                BufferedInputFile(merged.encode("utf-8"), filename=f"code_part{next_part}.{ext}"),
                caption=f"📄 Часть {next_part}",
                reply_markup=code_keyboard(project_id, is_complete=False)
            )
        await status.delete()
    except Exception as e:
        logging.error(f"CODE error: {e}")
        await status.edit_text(f"❌ {str(e)[:200]}")


async def continue_code_auto(msg, uid: int, project_id: str, next_part: int, ext: str):
    # ⏱ ПАУЗА между частями, чтобы не ловить 429
    if next_part > 1:
        await asyncio.sleep(10)
    
    try:
        old_parts = await get_code_parts(uid, project_id)
        old_code = merge_code_parts(old_parts, ext)
        meta = await get_code_meta(uid, project_id)
        original_request = meta.get('original_request', '')

        prompt = (
            f"ОРИГИНАЛЬНЫЙ ЗАПРОС: {original_request}\n\n"
            f"Продолжи код. Уже написано:\n\n```\n{old_code[-2500:]}\n```\n\n"
            f"ПИШИ ТОЛЬКО КОД. НЕ повторяй. НЕ разрывай строки. "
            f"Часть {next_part}. Максимум 800 токенов. "
            f"Маркер: `// (продолжение следует)` или `// (код готов)`."
        )
        r = await call_groq_with_retry(
            messages=[
                {"role": "system", "content": CODE_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200, temperature=0.5
        )
        answer = r.choices[0].message.content
        code = extract_code(answer)
        if not code or len(code.strip()) < 10:
            await msg.answer("⏸ Пустой ответ. Остановка.")
            return

        await save_code_part(uid, project_id, next_part, code, original_request[:100], original_request, ext)
        all_parts = await get_code_parts(uid, project_id)
        merged = merge_code_parts(all_parts, ext)
        is_complete = is_code_complete(answer)

        if is_complete:
            merged = finalize_code(merged, ext)
            await finish_code(uid, project_id)
            await msg.answer_document(
                BufferedInputFile(merged.encode("utf-8"), filename=f"final.{ext}"),
                caption=f"✅ Готово. Частей: {len(all_parts)}",
                reply_markup=code_keyboard(project_id, is_complete=True)
            )
        else:
            if next_part < MAX_AUTO_PARTS:
                await continue_code_auto(msg, uid, project_id, next_part + 1, ext)
            else:
                await msg.answer(f"⏸ Лимит {MAX_AUTO_PARTS} частей. Продолжи вручную.")
    except Exception as e:
        logging.error(f"auto error: {e}")
        await msg.answer(f"❌ {str(e)[:200]}")


async def continue_code(msg, uid: int, project_id: str):
    status = await msg.answer("💻 Дописываю...")
    try:
        old_parts = await get_code_parts(uid, project_id)
        old_code = "\n\n".join(old_parts)
        next_part = len(old_parts) + 1
        meta = await get_code_meta(uid, project_id)
        original_request = meta.get('original_request', '')
        ext = meta.get('file_ext', 'html')

        prompt = (
            f"ОРИГИНАЛЬНЫЙ ЗАПРОС: {original_request}\n\n"
            f"Продолжи код:\n\n```\n{old_code[-2500:]}\n```\n\n"
            f"ПИШИ ТОЛЬКО КОД. НЕ повторяй. НЕ разрывай строки. "
            f"Часть {next_part}. Максимум 800 токенов. "
            f"Маркер: `// (продолжение следует)` или `// (код готов)`."
        )
        r = await call_groq_with_retry(
            messages=[
                {"role": "system", "content": CODE_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200, temperature=0.5
        )
        answer = r.choices[0].message.content
        code = extract_code(answer)
        if not code or len(code.strip()) < 10:
            await status.edit_text("❌ Пустой ответ.")
            return
        await save_code_part(uid, project_id, next_part, code, original_request[:100], original_request, ext)
        all_parts = await get_code_parts(uid, project_id)
        merged = merge_code_parts(all_parts, ext)
        is_complete = is_code_complete(answer)
        if is_complete:
            merged = finalize_code(merged, ext)
            await msg.answer_document(
                BufferedInputFile(merged.encode("utf-8"), filename=f"final.{ext}"),
                caption=f"✅ Готово. Частей: {len(all_parts)}",
                reply_markup=code_keyboard(project_id, is_complete=True)
            )
        else:
            await msg.answer_document(
                BufferedInputFile(merged.encode("utf-8"), filename=f"code_part{next_part}.{ext}"),
                caption=f"📄 Часть {next_part}",
                reply_markup=code_keyboard(project_id, is_complete=False)
            )
        await status.delete()
    except Exception as e:
        logging.error(f"continue error: {e}")
        await status.edit_text(f"❌ {str(e)[:200]}")


# --- LLM-детект намерения ---
async def detect_code_intent(text: str) -> bool:
    try:
        r = await call_groq_with_retry(
            messages=[{"role": "user", "content":
                f"Пользователь написал: «{text}»\n"
                f"Хочет ли он написать код/игру/сайт/программу/бота? "
                f"Ответь ТОЛЬКО 'да' или 'нет'."}],
            max_tokens=5, temperature=0.1
        )
        return "да" in r.choices[0].message.content.strip().lower()
    except Exception as e:
        logging.error(f"[LLM-detect] {e}")
        return False


# --- Хендлеры ---
@dp.message(Command("start"))
async def start(msg: types.Message):
    await msg.answer(
        "Привет. Я код-бот.\n\n"
        "💻 Напиши что нужно — сгенерирую код файлом.\n"
        "🎤 Или отправь голосовое — распознаю и сделаю.\n"
        "📄 Скинь файл кода — доработаю.\n\n"
        "Пример: `Сделай игру змейка на HTML с уровнями и скинами`"
    )

@dp.message(Command("code"))
async def cmd_code(msg: types.Message):
    request = msg.text.replace("/code", "").strip()
    if not request:
        await msg.answer("💻 Что написать? `/code игра змейка на HTML`")
        return
    await make_code(msg, request)


@dp.callback_query(F.data.startswith("code_cont_"))
async def code_continue(cb: types.CallbackQuery):
    pid = cb.data.replace("code_cont_", "")
    await cb.answer("Продолжаю...")
    await continue_code(cb.message, cb.from_user.id, pid)

@dp.callback_query(F.data.startswith("code_done_"))
async def code_done(cb: types.CallbackQuery):
    pid = cb.data.replace("code_done_", "")
    await cb.answer("Завершаю...")
    parts = await get_code_parts(cb.from_user.id, pid)
    meta = await get_code_meta(cb.from_user.id, pid)
    ext = meta.get('file_ext', 'html')
    merged = finalize_code(merge_code_parts(parts, ext), ext)
    await finish_code(cb.from_user.id, pid)
    await cb.message.answer_document(
        BufferedInputFile(merged.encode("utf-8"), filename=f"final.{ext}"),
        caption=f"✅ Готово. Частей: {len(parts)}",
        reply_markup=code_keyboard(pid, is_complete=True)
    )

@dp.callback_query(F.data.startswith("code_auto_"))
async def code_auto(cb: types.CallbackQuery):
    pid = cb.data.replace("code_auto_", "")
    await cb.answer("⏩ Авто...")
    await cb.message.answer("⏩ Авто-режим: дописываю до конца. Паузы между частями — 10 сек.")
    parts = await get_code_parts(cb.from_user.id, pid)
    meta = await get_code_meta(cb.from_user.id, pid)
    ext = meta.get('file_ext', 'html')
    await continue_code_auto(cb.message, cb.from_user.id, pid, len(parts) + 1, ext)

@dp.callback_query(F.data.startswith("code_restart_"))
async def code_restart(cb: types.CallbackQuery):
    pid = cb.data.replace("code_restart_", "")
    await cb.answer("Начинаю заново...")
    await delete_code(cb.from_user.id, pid)
    await cb.message.answer("🔄 Заново. Напиши, что нужно.")


# --- Основной обработчик ---
@dp.message()
async def chat(msg: types.Message):
    uid = msg.from_user.id

    # 1. Голосовое
    if msg.voice:
        await bot.send_chat_action(msg.chat.id, "typing")
        try:
            text = await transcribe_audio(msg.voice.file_id, "ogg")
        except Exception as e:
            await msg.answer(f"❌ Ошибка распознавания: {str(e)[:150]}")
            return
        await msg.answer(f"🎤 Распознал: _{text}_")
        await make_code(msg, text)
        return

    # 2. Аудио
    if msg.audio:
        await bot.send_chat_action(msg.chat.id, "typing")
        ext = "mp3"
        if msg.audio.file_name and "." in msg.audio.file_name:
            ext = msg.audio.file_name.rsplit(".", 1)[-1].lower()
        try:
            text = await transcribe_audio(msg.audio.file_id, ext)
        except Exception as e:
            await msg.answer(f"❌ Ошибка: {str(e)[:150]}")
            return
        await msg.answer(f"🎤 Распознал: _{text}_")
        await make_code(msg, text)
        return

    # 3. Документ с кодом
    if msg.document:
        fname = msg.document.file_name or "file"
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        if ext not in CODE_EXTENSIONS:
            await msg.answer(f"❌ Поддерживаются: {', '.join(CODE_EXTENSIONS)}")
            return
        await bot.send_chat_action(msg.chat.id, "typing")
        try:
            f = await bot.get_file(msg.document.file_id)
            d = await bot.download_file(f.file_path)
            old_code = d.read().decode("utf-8", errors="ignore")
        except Exception as e:
            await msg.answer(f"❌ {str(e)[:150]}")
            return
        project_id = f"code_{uid}_{int(datetime.now().timestamp())}"
        await save_code_part(uid, project_id, 1, old_code, msg.caption or fname, msg.caption or fname, ext)
        prompt = (
            f"ОРИГИНАЛЬНЫЙ ЗАПРОС: {msg.caption or 'Доработай код'}\n\n"
            f"Вот код:\n\n```\n{old_code[-2500:]}\n```\n\n"
            f"Доработай или дополни. ПИШИ ТОЛЬКО КОД. "
            f"НЕ разрывай строки. Маркер: `// (продолжение следует)` или `// (код готов)`."
        )
        try:
            r = await call_groq_with_retry(
                messages=[{"role": "system", "content": CODE_PROMPT},
                          {"role": "user", "content": prompt}],
                max_tokens=1200, temperature=0.5
            )
            answer = r.choices[0].message.content
            new_code = extract_code(answer)
            merged = old_code.rstrip() + "\n\n" + new_code
            is_complete = is_code_complete(answer)
            if is_complete:
                merged = finalize_code(merged, ext)
            await msg.answer_document(
                BufferedInputFile(merged.encode("utf-8"), filename=f"updated.{ext}"),
                caption=f"{'✅ Готово' if is_complete else '📄 Продолжить?'}",
                reply_markup=code_keyboard(project_id, is_complete=False)
            )
        except Exception as e:
            await msg.answer(f"❌ {str(e)[:200]}")
        return

    # 4. Текст
    if msg.text:
        low = msg.text.lower()

        continue_triggers = ["допиши", "продолжи", "докончи", "дальше", "продолжай", "закончи"]
        if any(w in low for w in continue_triggers):
            active = await get_active_code(uid)
            if active:
                await continue_code(msg, uid, active['project_id'])
            else:
                await msg.answer("💻 Нет активного проекта. Напиши что сделать.")
            return

        fast_triggers = [
            "код", "игр", "сайт", "бот", "программ", "скрипт", "приложен",
            "html", "python", "js", "css", "java", "php", "sql",
            "создай", "сделай", "напиши", "хочу", "нужно", "сделать",
            "змейк", "динозавр", "тетрис", "арканоид", "2048", "flappy", "пятнашк"
        ]
        if any(w in low for w in fast_triggers):
            await make_code(msg, msg.text)
            return

        if await detect_code_intent(msg.text):
            await make_code(msg, msg.text)
            return

        await msg.answer(
            "💻 Я код-бот. Напиши что нужно сделать — сгенерирую код.\n"
            "Пример: `Сделай игру змейка на HTML с уровнями и скинами`"
        )


# --- Веб-сервер ---
async def handle(request):
    return web.Response(text="CodeBot is running!")

async def main():
    global db_pool
    logging.basicConfig(level=logging.INFO)
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    await init_db()
    logging.info("База данных подключена")

    app = web.Application()
    app.router.add_get("/", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logging.info(f"Web server port {port}")

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
