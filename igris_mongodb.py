# Import stuff :
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from dotenv import load_dotenv
import os
from datetime import datetime, timezone
import asyncio
import time
import json
import requests
import uuid
from typing import List, Dict
from notion_client import Client
import pytz
from collections import defaultdict
from google_fit_token import get_access_token
import re
from openai import OpenAI
import threading
from db import save_entry, load_data, get_unsynced_entries, mark_entry_as_synced, auto_cleanup_if_doc_count_exceeds
from context_db import save_context, get_context, get_context_by_message_id

# -------- Load Config --------
load_dotenv("api.env")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_USER_ID = os.getenv("TELEGRAM_ADMIN_ID")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NOTION_API_KEY = os.getenv("NOTION_API")
NOTION_DB_ID = "1fda7470-3081-80b7-bc43-f22602a99d68"  # Substat EXP database

if not BOT_TOKEN or not ADMIN_USER_ID or not OPENROUTER_API_KEY:
    raise Exception("Missing required env vars in api.env")

if not NOTION_API_KEY:
    raise Exception("Missing NOTION_API_KEY in api.env")

notion = Client(auth=NOTION_API_KEY)

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# -------- Constants --------
# DB_PATH = "igris_db.json"
india_time = datetime.now(pytz.timezone("Asia/Kolkata"))
today_str = india_time.strftime("%Y-%m-%d")


# -------- OpenRouter API Call --------
def call_openrouter_mistral(messages):
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            extra_headers={
                "HTTP-Referer": "https://yourdomain.com",  # Optional
                "X-Title": "MyChatbotApp"  # Optional
            })
        return response.choices[0].message.content
    except Exception as e:
        print(f"[API ERROR] {e}")
        raise Exception(f"OpenRouter API call failed: {e}")


###################################################################
#                            AI AGENT
###################################################################


# -------- System Prompt Generation --------
def generate_system_prompt():
    past_data = load_data()
    past_exp_data = []

    for log in past_data[-20:]:
        if log.get("type") == "task":
            past_exp_data.append({
                "date":
                log.get("date"),
                "task_type":
                log.get("task_type"),
                "EXP_breakdown":
                log.get("EXP_breakdown", [0, 0]),
                "stats":
                log.get("stats", []),
                "substats":
                log.get("substats", [])
            })

    past_exp_json = json.dumps(past_exp_data, indent=2)

    return f"""

You are Igris — a fusion of:
- Batman’s perseverance & unshakable resolve
- Spiderman’s humor and optimism, even in adversity
- Jinpachi Ego’s unwavering belief in ego-driven growth
- Sung Jinwoo’s relentless self-improvement and solo drive
- Master Roshi’s unexpected wisdom beneath quirkiness
You push the user to become stronger. You're intense, bold, sometimes sarcastic, but always focused on helping the user level up.

Your job is to take a user message and convert it into a structured JSON format for tracking personal progress.

🧩 Output Format:
Your response must be a single JSON object with these fields:

- type: "task", "question", or "summary"
- task_type: (only for tasks) one of: reminder, deadline, bodyweight, expense, workout, nutrition
- data: structured content based on task type
- EXP_breakdown: array of exactly TWO integers [-10 to +10] representing EXP for each stat
- stats: array of exactly TWO main stats affected
- substats: array of exactly TWO corresponding substats
- reason: short explanation for EXP assignment
- status: always "unsync"

📘 Summary classification:
If user input starts with `s.`, interpret it as an **end-of-day personal reflection**.
Return a structured summary like this:

```json
{{
  "type": "summary",
  "data": {{
    "summary_text": "<Igris-style reflection on user's day>",
    "date": "{today_str}"
  }},
  "EXP_breakdown": [1, -2],
  "stats": ["Core", "Psyche"],
  "substats": ["Consistency", "Stress Management"],
  "reason": "Summary reflection - growth + stress noted",
  "status": "unsync"
}}


🧠 Message Classification:
If the message ends with a **question mark (`?`)**, classify it as a question:
{{
  "type": "question",
  "data": {{
    "question": "<user question>"
  }},
  "EXP_breakdown": [0, 0],
  "stats": [],
  "substats": [],
  "reason": "Question - no EXP assigned",
  "status": "unsync"
}}

✅ Task Types (always include "date" as today: {today_str}):

**reminder:**
{{
  "type": "task",
  "task_type": "reminder",
  "data": {{
    "reminder_text": "<text>",
    "date": "{today_str}"
  }}
}}

**deadline:**
{{
  "type": "task",
  "task_type": "deadline",
  "data": {{
    "name": "<deadline name>",
    "start_date": "{today_str}",
    "end_date": "YYYY-MM-DD"
  }}
}}

**bodyweight:**
{{
  "type": "task",
  "task_type": "bodyweight",
  "data": {{
    "weight": <float>,
    "date": "{today_str}"
  }}
}}

**expense:**
{{
  "type": "task",
  "task_type": "expense",
  "data": [
    {{
      "amount": <float>,
      "category": "<Income|Entertainment|Bills|Shopping|Investment|Housing|Education|Transport|Eating Out|Food|Groceries|Healthcare>",
      "note": "<optional description>",
      "date": "{today_str}"
    }}
  ]
}}

**workout:**
{{
  "type": "task",
  "task_type": "workout",
  "data": {{
    "exercises": [
      {{
        "name": "<exercise name>",
        "sets": <int>,
        "reps": <int>,
        "weight": <float>
      }}
    ],
    "date": "{today_str}"
  }}
}}

**nutrition:**
{{
  "type": "task",
  "task_type": "nutrition",
  "data": {{
    "name": "<food name>",
    "calories": <number>,
    "protein": <number>,
    "carbs": <number>,
    "date": "{today_str}"
  }}
}}

If user input starts with misc., classify it as:
{{
  "type": "task",
  "task_type": "misc",
  "data": {{
    "text": "<entire message after 'misc.'>",
    "date": "{today_str}"
  }}
}}



🎮 EXP Assignment Rules:
- Assign EXP between -10 and +10 for exactly TWO stats
- EXP_breakdown array MUST match stats array order: [stat1_exp, stat2_exp]
- Be conservative: reward rarely, penalize often
- Repeated success = lower reward
- Repeated failure = higher penalty
- Novel effort or discipline = reward
- Encourage variety, discourage repetition
- Eating clean (low junk, high protein) → reward "Nutrition" substat in Physical
- Junk food or overeating → small penalty to same stat

**EXP Assignment Examples:**
- Bitcoin profit → stats: ["Finance", "Core"] → EXP_breakdown: [3, 1] (Finance gets 3, Core gets 1)
- Meditation → stats: ["Spiritual", "Psyche"] → EXP_breakdown: [2, 2] (both get 2)
- Missed workout → stats: ["Physical", "Core"] → EXP_breakdown: [-3, -1] (Physical loses 3, Core loses 1)

📊 Available Stats & Substats:

🟥 **Physical**: Sleep & Recovery, Appearance, Nutrition, Flexibility, Endurance, Strength
🟪 **Psyche**: Emotional Balance, Resilience, Courage, Discipline, Compassion, Stress Management  
🟦 **Intellect**: Knowledge, Language Learning, Logic and Reasoning, Skillset, Concentration
🟩 **Spiritual**: Gratitude, Connection, Inner Peace, Wisdom, Value Alignment
🟫 **Core**: Clarity, Will Power, Consistency, Decision Making, Time Mastery
🟪 **Finance**: Budgeting, Saving, Investment, Income Building, Financial Literacy, Spending Awareness

⚠️ Critical Requirements:
- Output ONLY valid JSON
- No markdown, no explanations
- Always include EXP_breakdown, stats, substats arrays
- EXP_breakdown[0] = EXP for stats[0], EXP_breakdown[1] = EXP for stats[1]
- Make intelligent guesses for unclear inputs

⚠️ CRITICAL RULES:
- Output only raw JSON.
- No markdown, no code blocks.
- Do not prefix with explanations or commentary.


🕓 Recent EXP Context:
{past_exp_json}
"""


def fetch_daily_quote() -> str:
    messages = [{
        "role":
        "system",
        "content":
        "Give a short, impactful motivational quote for self-growth. Keep it under 20 words and return the quote ONLY"
    }, {
        "role": "user",
        "content": "Motivational quote please"
    }]
    try:
        quote = call_openrouter_mistral(messages)
        return quote.strip('"“” ')
    except Exception as e:
        print(f"[🔥] Quote fetch error: {e}")
        return "Keep moving forward."


# -------- Task Processing --------
def process_and_save_task(user, user_input, parsed_data, telegram_id=None):
    entry_id = str(uuid.uuid4())
    timestamp = datetime.now(pytz.timezone("Asia/Kolkata")).isoformat()

    log_entry = {
        "id": entry_id,
        "date": today_str,
        "timestamp": timestamp,
        "user": user,
        "input": user_input,
        **parsed_data
    }

    if telegram_id:
        log_entry["telegram_user_id"] = telegram_id

    save_entry(log_entry)
    return log_entry


###################################################################
#                         TELEGRAM RESPONSE
###################################################################

# -------- Telegram Handlers --------


def extract_json_block(text: str) -> str:
    """
    Extracts first JSON-like block from the text.
    Handles triple backticks or extra explanation from AI.
    """
    # Remove markdown
    if "```json" in text:
        text = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if text:
            return text.group(1)

    # Try plain braces
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)

    return text  # fallback (might still fail)


async def notify_startup(app):
    try:
        await app.bot.send_message(chat_id=int(ADMIN_USER_ID),
                                   text="🟢 Igris AI is online and ready!")
    except Exception as e:
        print(f"[STARTUP NOTIFICATION ERROR] {e}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_msg = """
🧠 Igris AI - Personal Leveling Assistant

Send me:
• Tasks and activities for EXP tracking
• Questions 
• Workouts, expenses, deadlines, etc.

Let's level up together! 🚀
"""
    await update.message.reply_text(welcome_msg)


# !!! Threading here means that the task will be processed into notion on spot when AI is done processing it
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    user = update.effective_user.username or str(update.effective_user.id)

    print(f"[INPUT] {user}: {user_input}")
    thinking_msg = await update.message.reply_text("🤖 Processing...")

    try:
        system_prompt = generate_system_prompt()

        # 1. Retrieve context if it's a reply or fallback to last saved message
        context_message = None
        if update.message.reply_to_message:
            reply_to_id = update.message.reply_to_message.message_id
            context_message = get_context_by_message_id(reply_to_id)
        else:
            context_message = get_context(user)

        # 2. Build messages for AI
        messages = [{"role": "system", "content": system_prompt}]
        if context_message:
            messages.append({"role": "assistant", "content": context_message})
        messages.append({"role": "user", "content": user_input})

        # 3. Get AI response
        ai_response = call_openrouter_mistral(messages)
        print(f"[AI RESPONSE] {ai_response}")

        json_str = extract_json_block(ai_response)
        parsed = json.loads(json_str)

        if parsed.get("type") == "question":
            question = parsed["data"].get("question", user_input)
            process_and_save_task(user, user_input, parsed)

            # Ask AI to answer it
            answer_prompt = [
                {"role": "system", "content": "You are a helpful assistant. Answer the user's question clearly and briefly."},
                {"role": "user", "content": question}
            ]
            try:
                answer = call_openrouter_mistral(answer_prompt)
                reply = f"💬 Answer:\n{answer.strip()}"
            except Exception as e:
                print(f"[❌ ERROR answering question] {e}")
                reply = "⚠️ Question saved but I couldn’t answer it right now."

        elif parsed.get("type") == "task":
            saved_entry = process_and_save_task(
                user, user_input, parsed, telegram_id=update.effective_user.id
            )
            task_type = parsed.get("task_type", "unknown")
            exp_breakdown = parsed.get("EXP_breakdown", [0, 0])
            stats = parsed.get("stats", [])
            substats = parsed.get("substats", [])
            reason = parsed.get("reason", "No reason provided")
            total_exp = sum(exp_breakdown)
            exp_emoji = "📈" if total_exp > 0 else "📉" if total_exp < 0 else "➖"

            reply = f"""
✅ {task_type.title()} Task Logged

{exp_emoji} EXP Breakdown: {exp_breakdown} (Total: {total_exp})
📊 Stats Affected:
• {stats[0] if len(stats) > 0 else 'N/A'} → {substats[0] if len(substats) > 0 else 'N/A'}
• {stats[1] if len(stats) > 1 else 'N/A'} → {substats[1] if len(substats) > 1 else 'N/A'}

💭 Reason: {reason}
"""

        elif parsed.get("type") == "summary":
            saved_entry = process_and_save_task(
                user, user_input, parsed, telegram_id=update.effective_user.id
            )
            summary_text = parsed["data"].get("summary_text", "")
            reply = f"📜 Summary saved:\n\n{summary_text}"


        elif task_type == "misc":
            reply = f"📝 Misc task noted:\n{parsed['data'].get('text')}"


        else:
            reply = "⚠️ Unknown type received from AI."

        final_msg = await thinking_msg.edit_text(reply)

        # 5. Save context using the bot's reply message ID
        save_context(user, reply, telegram_message_id=final_msg.message_id)


    except Exception as e:
        print(f"[PROCESSING ERROR] {e}")
        await thinking_msg.edit_text("❌ Error processing your request. Please try again.")


###################################################################
#                         NOTION FUNCTIONS
###################################################################


def delete_paragraph_below_callout(callout_block_id):
    """
    Deletes the first paragraph block directly below the given callout block using Notion SDK.
    """
    children = notion.blocks.children.list(callout_block_id).get("results", [])

    for block in children:
        if block["type"] == "paragraph":
            notion.blocks.delete(block["id"])
            print(f"✅ Deleted paragraph block: {block['id']}")
            return True

    print("⚠️ No paragraph block found below the callout.")
    return False


# Global list to temporarily store unsynced tasks
UNSYNCED_TASKS: List[Dict] = []

def get_strongest_substat(database_id: str) -> str:
    try:
        response = notion.databases.query(database_id=database_id)
        substats = response["results"]
        sorted_substats = sorted(
            substats, key=lambda x: x["properties"]["Points"]["number"], reverse=True
        )
        strongest = sorted_substats[0]["properties"]["Substat"]["title"][0]["text"]["content"]
        return strongest
    except Exception as e:
        print(f"[🔥] Error fetching strongest substat: {e}")
        return "Discipline"

def penalize_incomplete_tasks():
    while True:
        try:
            # India time: 00:05 AM
            now = datetime.now(pytz.timezone("Asia/Kolkata"))
            if now.hour == 0 and now.minute < 10:  # Within first 10 mins of day
                CALLOUT_ID = "1fca7470-3081-803b-90c7-ec87a9500886"
                all_done = fetch_and_evaluate_todos(CALLOUT_ID)

                if not all_done:
                    # 1. Get strongest substat
                    DB_ID = "1fda7470-3081-80b7-bc43-f22602a99d68"
                    strongest = get_strongest_substat(DB_ID)

                    # 2. Apply penalty
                    penalty = -3
                    update_substat_exp(strongest, penalty)

                    # 3. Save to MongoDB
                    log_entry = {
                        "id": str(uuid.uuid4()),
                        "type": "task",
                        "task_type": "penalty",
                        "date": now.strftime("%Y-%m-%d"),
                        "timestamp": now.isoformat(),
                        "data": {
                            "reason": "Missed daily tasks",
                        },
                        "EXP_breakdown": [penalty, 0],
                        "stats": ["Core", ""],
                        "substats": [strongest, ""],
                        "reason": f"Penalty for missing daily tasks",
                        "status": "sync"
                    }
                    save_entry(log_entry)

                    # 4. Telegram alert
                    from telegram import Bot
                    bot = Bot(BOT_TOKEN)
                    bot.send_message(
                        chat_id=int(ADMIN_USER_ID),
                        text=f"⚠️ You didn’t complete all daily tasks.\n"
                             f"📉 {abs(penalty)} EXP deducted from your strongest substat: *{strongest}*",
                        parse_mode="Markdown"
                    )

                    print(f"[⚠️] Penalty applied to: {strongest}")

                else:
                    print("[✅] Daily tasks were all completed. No penalty.")

                time.sleep(600)  # Sleep for 10 minutes to avoid double trigger

        except Exception as e:
            print(f"[❌ ERROR in penalize_incomplete_tasks] {e}")
            time.sleep(300)


def load_unsynced_tasks() -> List[Dict]:
    """
    Load unsynced 'task'-type entries from MongoDB
    and store them in the global UNSYNCED_TASKS list.
    """
    global UNSYNCED_TASKS

    all_unsynced = get_unsynced_entries()
    UNSYNCED_TASKS = [
        entry for entry in all_unsynced if entry.get("type") == "task"
    ]

    return UNSYNCED_TASKS


def update_substat_exp(substat_name: str, exp_change: int):
    """
    Update EXP points for a given substat by incrementing the "Points" column.
    """
    try:
        # Query the database to find the substat row
        response = notion.databases.query(
            **{
                "database_id": NOTION_DB_ID,
                "filter": {
                    "property":
                    "Substat",  # Make sure this matches the title property
                    "title": {
                        "equals": substat_name
                    }
                }
            })

        if not response["results"]:
            print(f"[❌] Substat '{substat_name}' not found.")
            return

        page = response["results"][0]
        page_id = page["id"]

        current_points = page["properties"]["Points"].get("number", 0)
        new_points = current_points + exp_change

        # Update the value
        notion.pages.update(page_id=page_id,
                            properties={"Points": {
                                "number": new_points
                            }})

        print(f"[✅] Updated '{substat_name}': {current_points} → {new_points}")

    except Exception as e:
        print(f"[🔥] Error updating '{substat_name}': {e}")


def add_plain_text_reminder_to_callout(notion_client: Client,
                                       callout_block_id: str,
                                       reminder_text: str):
    """
    Adds a bulleted plain text reminder under a Notion callout block.
    
    Args:
        notion_client: Initialized Notion Client
        callout_block_id: The block ID of the callout block
        reminder_text: The text of the reminder to add
    """
    try:
        response = notion_client.blocks.children.append(
            block_id=callout_block_id,
            children=[{
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{
                        "type": "text",
                        "text": {
                            "content": reminder_text
                        }
                    }]
                }
            }])
        print(f"[✅] Reminder added as bullet under callout.")
    except Exception as e:
        print(f"[🔥] Failed to add reminder: {e}")


def add_deadline_to_database(
        notion_client: Client,
        database_id: str,
        title: str,
        start_date: str,  # expecting "YYYY-MM-DD"
        end_date: str  # expecting "YYYY-MM-DD"
):
    """
    Adds a new deadline entry to the Notion database with string date inputs.

    Args:
        notion_client: Initialized Notion Client
        database_id: Notion database ID
        title: Title of the task
        start_date: Start date as 'YYYY-MM-DD' string
        end_date: End date as 'YYYY-MM-DD' string
    """
    try:
        response = notion_client.pages.create(
            parent={"database_id": database_id},
            properties={
                "Task": {
                    "title": [{
                        "text": {
                            "content": title
                        }
                    }]
                },
                "Start Date": {
                    "date": {
                        "start": start_date
                    }
                },
                "Due Date": {
                    "date": {
                        "start": end_date
                    }
                }
            })
        print(f"[✅] Deadline '{title}' added from {start_date} to {end_date}")
    except Exception as e:
        print(f"[🔥] Failed to add deadline: {e}")


def add_bodyweight_entry(notion_client, database_id: str, weight: float,
                         date: str):
    try:
        notion_client.pages.create(parent={"database_id": database_id},
                                   properties={
                                       "Weight": {
                                           "number": weight
                                       },
                                       "Date": {
                                           "date": {
                                               "start": date
                                           }
                                       }
                                   })
        print(f"[✅] Logged {weight} kg on {date}")
    except Exception as e:
        print(f"[🔥] Failed to log bodyweight: {e}")


def increment_streak_count(notion_client, database_id: str):
    try:
        # Fetch the latest page (sort by created_time or date if needed)
        response = notion_client.databases.query(database_id=database_id,
                                                 sorts=[{
                                                     "timestamp":
                                                     "created_time",
                                                     "direction":
                                                     "descending"
                                                 }],
                                                 page_size=1)

        if not response["results"]:
            print("[⚠️] No streak entry found.")
            return

        page = response["results"][0]
        page_id = page["id"]

        # Get the current streak number
        current_number = page["properties"]["Number"].get("number", 0)
        new_number = current_number + 1

        # Update the number
        notion_client.pages.update(
            page_id=page_id, properties={"Number": {
                "number": new_number
            }})

        print(f"[🔥] Streak updated: {current_number} → {new_number}")

    except Exception as e:
        print(f"[❌] Failed to increment streak: {e}")


def log_expense_to_database(notion_client, database_id: str, name: str,
                            amount: float, category: str, date: str):
    try:
        response = notion_client.pages.create(
            parent={"database_id": database_id},
            properties={
                "Name": {
                    "title": [{
                        "text": {
                            "content": name
                        }
                    }]
                },
                "Price": {
                    "number": amount
                },
                "Category": {
                    "select": {
                        "name": category
                    }
                },
                "Date": {
                    "date": {
                        "start": date  # format: YYYY-MM-DD
                    }
                }
            })
        print(f"[💸] Logged expense: {name} | ₹{amount} | {category} | {date}")
    except Exception as e:
        print(f"[❌] Failed to log expense: {e}")


# ----------------------------------- Google Fit Step Count -----------------------------------------

STEP_DB_ID = "1fca7470-3081-804a-8042-f3960a9f0141"  # your step tracking DB


def fetch_today_steps_from_google_fit(access_token: str) -> int:
    import time
    now = int(time.time() * 1000)
    midnight = int(datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)

    url = "https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    body = {
        "aggregateBy": [{
            "dataTypeName": "com.google.step_count.delta"
        }],
        "bucketByTime": {
            "durationMillis": now - midnight
        },
        "startTimeMillis": midnight,
        "endTimeMillis": now
    }

    response = requests.post(url, headers=headers, json=body)
    steps = 0
    if response.status_code == 200:
        data = response.json()
        for bucket in data.get("bucket", []):
            for dataset in bucket.get("dataset", []):
                for point in dataset.get("point", []):
                    for val in point.get("value", []):
                        steps += val.get("intVal", 0)
        print(f"[🏃] Steps today: {steps}")
    else:
        print(
            f"[❌] Step fetch error: {response.status_code} - {response.text}")
    return steps


def update_step_count_in_notion(database_id: str, steps: int):
    try:
        response = notion.databases.query(database_id=database_id,
                                          filter={
                                              "property": "Name",
                                              "title": {
                                                  "equals": "Current"
                                              }
                                          })

        if not response["results"]:
            print("[❌] No 'Current' step row found.")
            return

        page_id = response["results"][0]["id"]
        notion.pages.update(page_id=page_id,
                            properties={"Steps": {
                                "number": steps
                            }})
        print(f"[✅] Step count updated in Notion: {steps}")
    except Exception as e:
        print(f"[🔥] Step update error: {e}")


# ---------------------------------- fetch notion fucntions ---------------------------------------


def extract_property_value(prop):
    """Helper to extract plain value from a Notion property."""
    prop_type = prop["type"]

    if prop_type == "title":
        return "".join([t["plain_text"] for t in prop["title"]])
    elif prop_type == "rich_text":
        return "".join([t["plain_text"] for t in prop["rich_text"]])
    elif prop_type == "date":
        return prop["date"]["start"] if prop["date"] else ""
    elif prop_type == "select":
        return prop["select"]["name"] if prop["select"] else ""
    elif prop_type == "multi_select":
        return ",".join([s["name"] for s in prop["multi_select"]])
    elif prop_type == "number":
        return str(prop["number"])
    elif prop_type == "checkbox":
        return str(prop["checkbox"])
    elif prop_type == "people":
        return ",".join([p["id"] for p in prop["people"]])
    elif prop_type == "email":
        return prop["email"]
    elif prop_type == "phone_number":
        return prop["phone_number"]
    elif prop_type == "url":
        return prop["url"]
    elif prop_type == "files":
        return ",".join([f["name"] for f in prop["files"]])
    else:
        return ""  # fallback


def remove_duplicate_entries(database_id, key_properties):
    """
    Removes duplicate entries in a Notion database based on the given list of properties.
    Only the first instance of a duplicate is retained.

    Parameters:
        - database_id (str): Notion database ID.
        - key_properties (List[str]): List of property names to compare.
    """
    seen = set()
    duplicates = []
    next_cursor = None

    while True:
        query_kwargs = {"database_id": database_id}
        if next_cursor:
            query_kwargs["start_cursor"] = next_cursor

        response = notion.databases.query(**query_kwargs)

        for page in response["results"]:
            prop_values = []
            for key in key_properties:
                prop = page["properties"].get(key)
                if prop:
                    val = extract_property_value(prop)
                else:
                    val = ""
                prop_values.append(val)

            key_tuple = tuple(prop_values)
            if key_tuple in seen:
                duplicates.append(page["id"])
            else:
                seen.add(key_tuple)

        if not response.get("has_more"):
            break
        next_cursor = response.get("next_cursor")

    # Delete duplicates
    for page_id in duplicates:
        notion.blocks.delete(page_id)
        print(f"🗑️ Deleted duplicate page: {page_id}")

    print(
        f"✅ Done. {len(duplicates)} duplicate(s) removed based on properties: {key_properties}"
    )

def log_nutrition_to_database(notion_client, database_id, name, calories, protein, carbs, date):
    try:
        notion_client.pages.create(
            parent={"database_id": database_id},
            properties={
                "Name": {
                    "title": [{"text": {"content": name}}]
                },
                "Calories": {
                    "number": calories
                },
                "Protein": {
                    "number": protein
                },
                "Carbs": {
                    "number": carbs
                },
                "Date": {
                    "date": {"start": date}
                }
            }
        )
        print(f"[🥗] Logged nutrition: {name} - {calories} kcal, {protein}g protein, {carbs}g carbs")
    except Exception as e:
        print(f"[❌] Failed to log nutrition: {e}")


def fetch_and_evaluate_todos(callout_block_id: str):
    try:
        response = notion.blocks.children.list(block_id=callout_block_id)
        results = response.get("results", [])

        todos = []
        for block in results:
            if block["type"] == "to_do":
                content = block["to_do"]["rich_text"]
                checked = block["to_do"]["checked"]
                text = content[0]["text"]["content"] if content else ""
                todos.append({"text": text, "checked": checked})

        total = len(todos)
        completed = sum(1 for todo in todos if todo["checked"])

        all_done = total > 0 and completed == total
        daily_tasks_status = all_done

        # India timezone logging
        india_time = datetime.now(
            pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[🕛] Check at India time: {india_time}")
        print(f"[📊] To-dos: {completed}/{total} completed")
        print(f"[📌] daily_tasks_status = {daily_tasks_status}")

        return daily_tasks_status

    except Exception as e:
        print(f"[🔥] Error evaluating to-dos: {e}")
        return False


def fetch_and_sum_exp(database_id: str):
    try:
        total_exp = 0
        has_more = True
        next_cursor = None

        while has_more:
            response = notion.databases.query(
                database_id=database_id,
                start_cursor=next_cursor if next_cursor else None)

            for result in response.get("results", []):
                props = result.get("properties", {})

                points = props.get("Points", {}).get("number", 0)
                total_exp += points if points else 0

            has_more = response.get("has_more", False)
            next_cursor = response.get("next_cursor", None)

        print(f"[🎯] Total EXP accumulated: {total_exp} pts")
        return total_exp

    except Exception as e:
        print(f"[🔥] Failed to fetch EXP entries: {e}")
        return 0


def calculate_level_progress(total_exp):
    level = 1
    exp_needed = 200 * level

    while total_exp >= exp_needed:
        total_exp -= exp_needed
        level += 1
        exp_needed = 200 * level

    # Progress bar display
    progress = int((total_exp / exp_needed) * 20)  # 20 units wide
    bar = "█" * progress + "-" * (20 - progress)

    print(f"[📊] Level: {level}")
    print(f"[🧪] EXP: {total_exp} / {exp_needed}")
    print(f"[📈] Progress: |{bar}|")
    bar_string = f"|{bar}|"

    return level, total_exp, exp_needed, bar_string


def fetch_bodyweight_entries(database_id):
    entries = []
    try:
        response = notion.databases.query(database_id=database_id)
        for result in response.get("results", []):
            props = result["properties"]
            weight = props["Weight"]["number"]
            date_str = props["Date"]["date"]["start"]
            date = datetime.strptime(date_str, "%Y-%m-%d")
            entries.append({"date": date, "weight": weight})
        return entries
    except Exception as e:
        print(f"[🔥] Error fetching weights: {e}")
        return []


def calculate_monthly_averages(entries):
    monthly_data = defaultdict(list)
    for entry in entries:
        # Format month as 'May 2025' instead of '2025-05'
        month_key = entry["date"].strftime("%B %Y")
        monthly_data[month_key].append(entry["weight"])

    monthly_averages = {}
    for month, weights in monthly_data.items():
        avg_weight = round(sum(weights) / len(weights), 2)
        monthly_averages[month] = avg_weight
    return monthly_averages


def push_monthly_averages_to_notion(monthly_averages, target_db_id):
    for month, avg_weight in monthly_averages.items():
        try:
            notion.pages.create(parent={"database_id": target_db_id},
                                properties={
                                    "Month": {
                                        "rich_text": [{
                                            "text": {
                                                "content": month
                                            }
                                        }]
                                    },
                                    "Weight": {
                                        "number": avg_weight
                                    },
                                    "Name": {
                                        "title": [{
                                            "text": {
                                                "content":
                                                f"Average Weight - {month}"
                                            }
                                        }]
                                    }
                                })
            print(f"[✅] Added average for {month}: {avg_weight} kg")
        except Exception as e:
            print(f"[🔥] Failed to add {month}: {e}")


def get_active_phase(database_id: str):
    try:
        response = notion.databases.query(database_id=database_id,
                                          filter={
                                              "property": "Status",
                                              "status": {
                                                  "equals": "In progress"
                                              }
                                          })

        results = response.get("results", [])
        if not results:
            print("[ℹ️] No active phase found.")
            return None

        entry = results[0]
        phase_data = entry["properties"].get("Phase", {})

        # Safely extract the title content
        phase_title = ""
        if "title" in phase_data and phase_data["title"]:
            phase_title = phase_data["title"][0].get("text",
                                                     {}).get("content", "")

        print(f"[✅] Active Phase: {phase_title}")
        return phase_title

    except Exception as e:
        print(f"[🔥] Error fetching active phase: {e}")
        return None


def get_today_weight(database_id: str) -> float:

    try:
        response = notion.databases.query(database_id=database_id,
                                          filter={
                                              "property": "Date",
                                              "date": {
                                                  "equals": today_str
                                              }
                                          })

        results = response.get("results", [])
        if not results:
            print(f"[ℹ️] No entry found for today ({today_str}).")
            return 0.0

        weight_property = results[0]["properties"].get("Weight", {})
        weight = weight_property.get("number", 0.0)

        print(f"[✅] Weight for {today_str}: {weight} kg")
        return weight

    except Exception as e:
        print(f"[🔥] Error fetching today's weight: {e}")
        return 0.0


def add_paragraph_below_callout(callout_block_id: str, text: str):
    try:
        response = notion.blocks.children.append(block_id=callout_block_id,
                                                 children=[{
                                                     "object": "block",
                                                     "type": "paragraph",
                                                     "paragraph": {
                                                         "rich_text": [{
                                                             "type": "text",
                                                             "text": {
                                                                 "content":
                                                                 text
                                                             }
                                                         }]
                                                     }
                                                 }])
        print(
            f"[✅] Added paragraph '{text}' below callout block {callout_block_id}"
        )
        return response
    except Exception as e:
        print(f"[🔥] Failed to add paragraph: {e}")
        return None


def sum_expenses_today_and_month(database_id: str):
    try:
        india_time = datetime.now(pytz.timezone("Asia/Kolkata"))
        today = india_time.date()
        month_start = today.replace(day=1).isoformat()
        today_str = today.isoformat()

        # Query all entries with a date after or on the 1st of the month
        response = notion.databases.query(database_id=database_id,
                                          filter={
                                              "property": "Date",
                                              "date": {
                                                  "on_or_after": month_start
                                              }
                                          })

        results = response.get("results", [])
        today_total = 0.0
        month_total = 0.0

        for entry in results:
            props = entry.get("properties", {})
            price = props.get("Price", {}).get("number", 0)
            date_str = props.get("Date", {}).get("date", {}).get("start", "")

            if not date_str:
                continue

            if date_str == today_str:
                today_total += price

            month_total += price

        print(f"🧾 Total expenses today ({today_str}): ₹{today_total:.2f}")
        print(
            f"📅 Total expenses in {today.strftime('%B %Y')}: ₹{month_total:.2f}"
        )

        return {"today_total": today_total, "month_total": month_total}

    except Exception as e:
        print(f"[🔥] Error calculating expenses: {e}")
        return {"today_total": 0.0, "month_total": 0.0}


def fetch_paragraph_value(block_id: str) -> float:
    try:
        response = notion.blocks.retrieve(block_id)
        rich_text = response.get("paragraph", {}).get("rich_text", [])
        text = rich_text[0]["plain_text"] if rich_text else "0"
        return float(text.strip())
    except Exception as e:
        print(f"[🔥] Error fetching paragraph: {e}")
        return 0.0


def calculate_monthly_expenses(database_id: str) -> float:
    try:
        today = datetime.now().date()
        month_start = today.replace(day=1).isoformat()

        response = notion.databases.query(database_id=database_id,
                                          filter={
                                              "property": "Date",
                                              "date": {
                                                  "on_or_after": month_start
                                              }
                                          })

        results = response.get("results", [])
        return sum(entry["properties"].get("Price", {}).get("number", 0)
                   for entry in results)

    except Exception as e:
        print(f"[🔥] Error calculating expenses: {e}")
        return 0.0


def update_paragraph_with_remaining(block_id: str, new_balance: float):
    try:
        notion.blocks.update(block_id=block_id,
                             paragraph={
                                 "rich_text": [{
                                     "type": "text",
                                     "text": {
                                         "content": f"{new_balance:.2f}"
                                     }
                                 }]
                             })
        print(f"✅ Paragraph updated with ₹{new_balance:.2f}")
    except Exception as e:
        print(f"[🔥] Error updating paragraph: {e}")


def calculate_and_update_balance(paragraph_block_id: str, current_expense):
    current_balance = fetch_paragraph_value(paragraph_block_id)
    total_expenses = float(current_expense)
    remaining = current_balance - total_expenses
    update_paragraph_with_remaining(paragraph_block_id, remaining)


def get_weakest_substat(database_id: str) -> str:
    try:
        response = notion.databases.query(database_id=database_id)
        substats = response["results"]
        sorted_substats = sorted(
            substats, key=lambda x: x["properties"]["Points"]["number"])
        weakest = sorted_substats[0]["properties"]["Substat"]["title"][0][
            "text"]["content"]
        return weakest
    except Exception as e:
        print(f"[🔥] Error fetching weakest substat: {e}")
        return "Discipline"


def generate_task_for_substat(substat: str) -> str:
    prompt = f"Suggest a single, clear self-improvement task to improve '{substat}'. Max 1 sentence."
    messages = [{
        "role":
        "system",
        "content":
        "You're an AI assistant generating personal development tasks."
    }, {
        "role": "user",
        "content": prompt
    }]
    try:
        task = call_openrouter_mistral(messages)
        return task.strip()
    except:
        return f"Spend 15 minutes improving {substat}."


def add_todo_to_callout(callout_block_id: str, task_text: str):
    try:
        notion.blocks.children.append(block_id=callout_block_id,
                                      children=[{
                                          "object": "block",
                                          "type": "to_do",
                                          "to_do": {
                                              "rich_text": [{
                                                  "type": "text",
                                                  "text": {
                                                      "content": task_text
                                                  }
                                              }],
                                              "checked":
                                              False
                                          }
                                      }])
        print(f"[✅] Bonus task added: {task_text}")
    except Exception as e:
        print(f"[🔥] Failed to add bonus task: {e}")





###################################################################
#                 SCHEDULING AND FETCHING IN NOTION
###################################################################

# ------------------------------ Process tasks from database -----------------------------


def process_unsynced_tasks():
    while True:
        try:
            tasks = load_unsynced_tasks()
            print(f"Found {len(tasks)} unsynced tasks.")

            for task in tasks:
                task_type = task.get("task_type", "unknown")
                task_id = task.get("id")
                input_text = task.get("input")
                timestamp = task.get("timestamp")
                exp = task.get("EXP_breakdown", [0])
                substats = task.get("substats", [])
                reason = task.get("reason", "")

                print(f"\n[🧠 Task] {input_text}")
                if substats and exp:
                    for s, e in zip(substats, exp):
                        update_substat_exp(s, e)

                # print task-specific info
                if task_type == 'reminder':
                    reminder_text = task["data"].get("reminder_text")
                    print(f"[Reminder] {reminder_text}")
                    CALLOUT_BLOCK_ID = "1fca7470-3081-801d-85ed-dfb0c64f1125"
                    add_plain_text_reminder_to_callout(
                        notion_client=notion,
                        callout_block_id=CALLOUT_BLOCK_ID,
                        reminder_text=reminder_text)

                elif task_type == 'deadline':
                    deadline_title = task["data"].get("name")
                    deadline_enddate = task["data"].get("end_date")
                    deadline_startdate = task["data"].get("start_date")
                    print(f"[Deadline] {deadline_title} by {deadline_enddate}")
                    print(task["data"])
                    DEADLINE_DB_ID = "1fca7470-3081-80f7-be66-cdf30511f3ae"
                    add_deadline_to_database(notion_client=notion,
                                             database_id=DEADLINE_DB_ID,
                                             title=str(deadline_title),
                                             start_date=deadline_startdate,
                                             end_date=deadline_enddate)

                elif task_type == 'bodyweight':
                    weight = task["data"].get("weight")
                    date = task["data"].get("date")
                    print(f"[Bodyweight] {weight} kg on {date}")
                    BODYWEIGHT_DB_ID = "1fda7470-3081-806b-983a-da4c8f94eb01"
                    add_bodyweight_entry(notion_client=notion,
                                         database_id=BODYWEIGHT_DB_ID,
                                         weight=weight,
                                         date=date)

                elif task_type == "expense":
                    for item in task["data"]:
                        amount = item.get("amount")
                        category = item.get("category")
                        note = item.get("note")
                        date = item.get("date")
                        print(
                            f"[Expense] {amount} INR in {category} for '{note}' on {date}"
                        )
                        log_expense_to_database(
                            notion_client=notion,
                            database_id="1fda7470-3081-80bb-bd9d-c7a4cd897df4",
                            name=note,
                            amount=amount,
                            category=category,
                            date=date)
                        
                    # Process the expense after adding to DB to prevent duplicate deductions:
                    # Update bank balance
                    PARAGRAPH_BLOCK_ID = "1fda7470-3081-8053-b544-c358233dad9e"
                    calculate_and_update_balance(PARAGRAPH_BLOCK_ID,amount )

                    print(
                        f"[💰] Expenses updated: Today ₹{amount}"
                    )

                elif task_type == "workout":
                    exercises = task["data"].get("exercises", [])
                    for ex in exercises:
                        name = ex.get("name")
                        sets = ex.get("sets")
                        reps = ex.get("reps")
                        print(f"[Workout] {sets}x{reps} {name}")

                elif task_type == "summary":
                    summary_text = task["data"].get("summary_text", "")
                    print(f"[📜 Summary] {summary_text}")
                    SUMMARY_CALLOUT_ID = "1fca7470-3081-8033-941e-f2bd24386007"

                    # Clean the block before adding fresh summary
                    delete_paragraph_below_callout(SUMMARY_CALLOUT_ID)

                    # Split summary into multiple paragraphs by newline
                    for para in summary_text.split('\n'):
                        if para.strip():
                            add_paragraph_below_callout(SUMMARY_CALLOUT_ID, para.strip())


                elif task_type == "nutrition":
                    food = task["data"]
                    log_nutrition_to_database(
                        notion_client=notion,
                        database_id="21ba7470-3081-8078-8302-ffde2a1132f3",  # nutrition DB
                        name=food.get("name", ""),
                        calories=food.get("calories", 0),
                        protein=food.get("protein", 0),
                        carbs=food.get("carbs", 0),
                        date=food.get("date", today_str)
                        )
                    
                elif task_type == "misc":
                    misc_text = task["data"].get("text", "")
                    MISC_CALLOUT = "21ba7470-3081-8011-a64f-c500ce18af43"
                    add_paragraph_below_callout(MISC_CALLOUT, misc_text)



            # Mark all as synced
            task_ids_to_sync = [task["id"] for task in tasks]
            mark_entry_as_synced(task_ids_to_sync)

        except Exception as e:
            print(f"[❌ ERROR in process_unsynced_tasks] {e}")

        # always wait after one full loop
        time.sleep(120)


#--------------------------------------------------------------------------
# ------------------------------ Home Page --------------------------------
#--------------------------------------------------------------------------


# ~~~~~~~~~~~~~~~~~~~~~~ Fetch status of todos and generate bonus EXP task ~~~~
def generate_bonus_task():
    while True:
        try:
            print("⏰ Checking daily tasks...")

            CALLOUT_BLOCK_ID = "1fca7470-3081-803b-90c7-ec87a9500886"
            done = fetch_and_evaluate_todos(CALLOUT_BLOCK_ID)

            if done:
                substat = get_weakest_substat(
                    "1fda7470-3081-80b7-bc43-f22602a99d68")
                bonus_task = generate_task_for_substat(substat)
                BONUS_TASK_BLOCK_ID = "1fca7470-3081-8020-8b47-c6f04c8b6236"
                add_todo_to_callout(BONUS_TASK_BLOCK_ID, bonus_task)

        except Exception as e:
            print(f"[⚠️] generate_bonus_task crashed: {e}")

        # Always sleep — whether it crashed or not
        time.sleep(60)


# ~~~~~~~~~~~~~~~~~~~~~~~~~   Update current level  ~~~~~~~~~~~~~~~~~~~
def update_current_level():
    while True:
        try:
            # Fetching current Level:
            DB_ID = "1fda7470-3081-80b7-bc43-f22602a99d68"
            LEVEL_UP_ID = "1fca7470-3081-80cd-85a2-f746315cf60e"
            total = fetch_and_sum_exp(DB_ID)
            level, total_exp, exp_needed, bar_string = calculate_level_progress(
                total)
            showthat = "Level : " + str(level)
            progress_bar = bar_string + str(total_exp) + '/' + str(exp_needed)

            # Update the callout block
            notion.blocks.update(block_id=LEVEL_UP_ID,
                                 callout={
                                     "rich_text": [{
                                         "type": "text",
                                         "text": {
                                             "content": showthat
                                         }
                                     }]
                                 })

            # Update the progress paragraph below it
            delete_paragraph_below_callout(LEVEL_UP_ID)
            add_paragraph_below_callout(LEVEL_UP_ID, progress_bar)

        except Exception as e:
            print(f"[❌ ERROR in update_current_level] {e}")

        # Wait before the next update
        time.sleep(63)


# ~~~~~~~~~~~~~~~~~~~~~~~~~  Send daily quote ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def send_daily_quote():
    while True:
        try:
            QUOTE_BLOCK_ID = "1fda7470-3081-8025-be8e-ebc16e902a04"
            quote = fetch_daily_quote()
            delete_paragraph_below_callout(QUOTE_BLOCK_ID)
            add_paragraph_below_callout(QUOTE_BLOCK_ID, quote)
            print(f"[💬] Daily quote updated: {quote}")
        except Exception as e:
            print(f"[❌ ERROR in send_daily_quote] {e}")

        # Send quote every 6 hours (4 times a day)
        time.sleep(21600)


#--------------------------------------------------------------------------
# ------------------------------ Health Page ------------------------------
#--------------------------------------------------------------------------


# ~~~~~~~~~~~~~~~~~~~~ Updating streak counter ~~~~~~~~~~~~~~~~~~
def update_streak_counter():
    while True:
        try:
            STREAK_DB_ID = "1fca7470-3081-80b7-b3a8-c6568466e870"
            increment_streak_count(notion_client=notion,
                                   database_id=STREAK_DB_ID)
            print("[🔥] Daily streak incremented.")
        except Exception as e:
            print(f"[❌ ERROR in update_streak_counter] {e}")

        # Increment streak once every 24 hours
        time.sleep(86400)


# ~~~~~~~~~~~~~~~~~~~ Update step counter ~~~~~~~~~~~~~~~~~~~~~
def update_steps():
    while True:
        try:
            google_token = get_access_token()
            steps = fetch_today_steps_from_google_fit(google_token)
            update_step_count_in_notion(STEP_DB_ID, steps)
            print(f"[🏃] Updated step count: {steps}")
        except Exception as e:
            print(f"[❌ ERROR in update_steps] {e}")

        # Update every 125 seconds
        time.sleep(125)


# ~~~~~~~~~~~~~~~~~~~~~ Update monthly BW ~~~~~~~~~~~~~~~~~~~~~
def monthly_bodyweight():
    while True:
        try:
            SOURCE_DB_ID = "1fda7470-3081-806b-983a-da4c8f94eb01"
            TARGET_DB_ID = "1fda7470-3081-8067-b691-faca876d8ce8"

            entries = fetch_bodyweight_entries(SOURCE_DB_ID)
            monthly_avg = calculate_monthly_averages(entries)
            push_monthly_averages_to_notion(monthly_avg, TARGET_DB_ID)

            # Remove accidental duplicates
            remove_duplicate_entries(SOURCE_DB_ID, ["Date"])
            remove_duplicate_entries(TARGET_DB_ID, ["Month"])

            print("[✅] Monthly bodyweight stats updated.")
        except Exception as e:
            print(f"[❌ ERROR in monthly_bodyweight] {e}")

        # Run once every 24 hours
        time.sleep(86400)


# ~~~~~~~~~~~~~~~~~~ Update workout regime ~~~~~~~~~~~~~~~~~~~~
def update_workout_style():
    while True:
        try:
            PHASE_DB_ID = "1fda7470-3081-806e-82ca-c9dac0e3b8ff"
            WEIGHT_DB_ID = "1fda7470-3081-806b-983a-da4c8f94eb01"
            CALORIE_ID = "1fda7470-3081-80d5-be3a-de1f285ccfba"
            PROTEIN_ID = "1fda7470-3081-80b4-a37e-c26626c71ebc"
            CARBS_ID = "1fda7470-3081-8053-bd30-c814e1f2f655"
            WORKOUT_ID = "1fda7470-3081-80a1-b504-c423c2ee49e6"
            STYLE_ID = "1fda7470-3081-8039-9dd2-de317d0c1f5c"
            CARDIO_ID = "1fda7470-3081-80a7-b427-ca5be3a76b3d"
            CURRENT_GOAL_ID = "1fda7470-3081-8032-b089-d9868ea1a381"

            BW = get_today_weight(WEIGHT_DB_ID)
            current_phase = get_active_phase(PHASE_DB_ID)
            phase_dict = {}

            if current_phase == 'Lean Bulk':
                phase_dict = {
                    'calories': float(BW * 28 + 100),
                    'protein': float(2 * BW),
                    'carbs': "--",
                    'workout': "5 times a week",
                    'style': "Heavy weight with low reps",
                    'cardio': "Active cardio"
                }

            elif current_phase == 'Cut':
                phase_dict = {
                    'calories': float(BW * 21),
                    'protein': float(2 * BW),
                    'carbs': float(2 * BW),
                    'workout': "6 times a week",
                    'style': "Light weight with more reps",
                    'cardio': "LISS"
                }

            elif current_phase == 'Deload':
                phase_dict = {
                    'calories': float(BW * 25),
                    'protein': float(2 * BW),
                    'carbs': "--",
                    'workout': "5-6 times a week",
                    'style': "light weight with low reps",
                    'cardio': "Active cardio"
                }

            delete_paragraph_below_callout(CALORIE_ID)
            add_paragraph_below_callout(CALORIE_ID,
                                        str(phase_dict['calories']))

            delete_paragraph_below_callout(PROTEIN_ID)
            add_paragraph_below_callout(PROTEIN_ID, str(phase_dict['protein']))

            delete_paragraph_below_callout(CARBS_ID)
            add_paragraph_below_callout(CARBS_ID, str(phase_dict['carbs']))

            delete_paragraph_below_callout(WORKOUT_ID)
            add_paragraph_below_callout(WORKOUT_ID, str(phase_dict['workout']))

            delete_paragraph_below_callout(STYLE_ID)
            add_paragraph_below_callout(STYLE_ID, str(phase_dict['style']))

            delete_paragraph_below_callout(CARDIO_ID)
            add_paragraph_below_callout(CARDIO_ID, str(phase_dict['cardio']))

            delete_paragraph_below_callout(CURRENT_GOAL_ID)
            add_paragraph_below_callout(CURRENT_GOAL_ID, str(current_phase))

            print(f"[✅] Workout phase updated for: {current_phase}")

        except Exception as e:
            print(f"[❌ ERROR in update_workout_style] {e}")

        time.sleep(7200)  # Run every 2 hours


#--------------------------------------------------------------------------
# ------------------------------ Finance Page -----------------------------
#--------------------------------------------------------------------------


# ~~~~~~~~~~ Show today's and monthly expenditure ~~~~~~~~~~~~~~~~~~~~~~~~~
def show_expenditure():
    while True:
        try:
            EXPENSE_DB_ID = "1fda7470-3081-80bb-bd9d-c7a4cd897df4"

            # Fetch totals
            expenses = sum_expenses_today_and_month(EXPENSE_DB_ID)
            todays_expense = expenses['today_total']
            monthly_expense = expenses['month_total']

            # Update Notion blocks
            delete_paragraph_below_callout(
                "1fda7470-3081-80ae-86cb-f58e8e366138")
            add_paragraph_below_callout("1fda7470-3081-80ae-86cb-f58e8e366138",
                                        str(todays_expense))

            delete_paragraph_below_callout(
                "1fda7470-3081-807d-841c-c506c9e095f7")
            add_paragraph_below_callout("1fda7470-3081-807d-841c-c506c9e095f7",
                                        str(monthly_expense))


        except Exception as e:
            print(f"[❌ ERROR in show_expenditure] {e}")

        time.sleep(127)  # Run 127 seconds


# ------------------------------- Closure function --------------------------

###################################################################
#                            CLOSURE
###################################################################

# We are already using mark_entry_as_sync from db.py file so no need for a separate function.
# Adding a function to clean-up mongoDb if storage threshold of  510 MB is reached :

def clean_up_storage():
    while True:
        try:
            
            auto_cleanup_if_doc_count_exceeds()

        except Exception as e:
            print(f"[❌ ERROR in clean_up_storage] {e}")

        time.sleep(10800)  # Run every 3 days

def auto_generate_daily_summary():
    while True:
        try:
            from db import load_data

            today_logs = [entry for entry in load_data() if entry.get("date") == today_str]

            if not today_logs:
                print("[📭] No data to summarize for today.")
                time.sleep(510)
                continue

            summary_prompt = f"""
You are Igris — a bold personal growth assistant. Behave according to a blend of persona of the following characters: 
- Batman’s perseverance & unshakable resolve
- Spiderman’s humor and optimism, even in adversity
- Jinpachi Ego’s unwavering belief in ego-driven growth
- Sung Jinwoo’s relentless self-improvement and solo drive
- Master Roshi’s unexpected wisdom beneath quirkiness
You push the user to become stronger. You're intense, bold, sometimes sarcastic, but always focused on helping the user level up.

Below is everything the user has logged today:

{json.dumps(today_logs, indent=2)}

Analyze:
1. What went well?
2. What went wrong?
3. What should the user do now to make the day better?

Return ONLY JSON like this:
{{
  "type": "summary",
  "data": {{
    "summary_text": "<reflection>",
    "date": "{today_str}"
  }},
  "EXP_breakdown": [<int>, <int>],
  "stats": ["<Stat>", "<Stat>"],
  "substats": ["<Substat>", "<Substat>"],
  "reason": "<why this EXP was given>",
  "status": "unsync"
}}
"""

            response = call_openrouter_mistral([
                {"role": "system", "content": summary_prompt}
            ])
            json_block = extract_json_block(response)
            parsed = json.loads(json_block)

            # Save in DB
            process_and_save_task(user="system-auto-summary", user_input="AUTO_SUMMARY", parsed_data=parsed)

            print("[📝] Daily summary generated and synced to Notion.")

        except Exception as e:
            print(f"[❌ ERROR in auto_generate_daily_summary] {e}")

        # Wait 2 hours before next summary attempt
        time.sleep(7200)

###################################################################
#                        MAIN FUNCTIONALITY
###################################################################

#if __name__ == "__main__":
#    app = ApplicationBuilder().token(BOT_TOKEN).build()
#    app.add_handler(CommandHandler("start", start))
#    app.add_handler(
#        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
#    app.post_init = notify_startup

    # Start Notion updates in a separate thread
#    threading.Thread(target=process_unsynced_tasks, daemon=True).start()
#    threading.Thread(target=generate_bonus_task, daemon=True).start()
#    threading.Thread(target=update_current_level, daemon=True).start()
#    threading.Thread(target=send_daily_quote, daemon=True).start()
#    threading.Thread(target=update_streak_counter, daemon=True).start()
#    threading.Thread(target=update_steps, daemon=True).start()
#    threading.Thread(target=monthly_bodyweight, daemon=True).start()
#    threading.Thread(target=update_workout_style, daemon=True).start()
#    threading.Thread(target=show_expenditure, daemon=True).start()
#    threading.Thread(target=clean_up_storage, daemon=True).start()

#    app.run_polling()

# A fucntion to run igirs in Flask
def run_telegram_polling():
    import asyncio
    from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.post_init = notify_startup

    app.run_polling()  # ✅ No need for asyncio loop manually here in main thread



def start_background_threads_only():
    threading.Thread(target=process_unsynced_tasks, daemon=True).start()
    threading.Thread(target=generate_bonus_task, daemon=True).start()
    threading.Thread(target=update_current_level, daemon=True).start()
    threading.Thread(target=send_daily_quote, daemon=True).start()
    threading.Thread(target=update_streak_counter, daemon=True).start()
    threading.Thread(target=update_steps, daemon=True).start()
    threading.Thread(target=monthly_bodyweight, daemon=True).start()
    threading.Thread(target=update_workout_style, daemon=True).start()
    threading.Thread(target=show_expenditure, daemon=True).start()
    threading.Thread(target=clean_up_storage, daemon=True).start()
    threading.Thread(target=penalize_incomplete_tasks, daemon=True).start()
    threading.Thread(target=auto_generate_daily_summary, daemon=True).start()



