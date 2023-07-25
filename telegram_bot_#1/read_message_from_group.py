
from telegram.ext import Application, MessageHandler, filters, CallbackContext
from telegram import Update
from config import Token
group_chat_id = -1001582794605,  # Replace with the ID of your group chat


async def handle_messages(update: Update, context: CallbackContext) -> None:
    # Get the message text and chat ID
    message_text = update.message.text
    # print("Check:", update.message)
    chat_id = update.message.chat_id 
    # Print the message to the console
    print(f"Received message '{message_text}' from chat ID {chat_id}")

  
application = Application.builder().token(Token).build()
group_message_handler = MessageHandler(filters.Chat(chat_id=group_chat_id) & filters.TEXT, handle_messages)
application.add_handler(group_message_handler)
application.run_polling()