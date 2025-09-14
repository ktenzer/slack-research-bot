import sys
import os
import re
import asyncio

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

relative_path = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(relative_path)

from tools import (
    get_slack_channels,
    search_slack,
    get_thread_messages,
    PromptUser,
    send_slack_message,
)
from sys_prompt import get_system_prompt
from libs.agent.agent import Agent

async def main():
    # Slack credentials
    bot_token  = os.environ["SLACK_BOT_TOKEN"]
    app_token  = os.environ["SLACK_APP_TOKEN"]

    # Initialise the LLM-powered agent
    async with Agent(
        model_name="gpt-4o-mini",
        instruction=get_system_prompt(),
        functions=[
            get_slack_channels,
            search_slack,
            get_thread_messages,
            PromptUser,
            send_slack_message,
        ],
    ) as agent:
        slack_app = AsyncApp(token=bot_token)  

        # Handle mentions
        @slack_app.event("app_mention")
        async def handle_app_mention(body, say):
            """
            Whenever the bot is mentioned, strip the mention from the text,
            send it to the agent, and post the reply back to Slack.
            """
            raw_text = body["event"]["text"]
            # Remove only the leading bot mention (e.g. "<@U123ABC> hello" → "hello")
            user_text = re.sub(r"^<@[^>]+>\s*", "", raw_text).strip()
            if not user_text:
                await say("Please include a question after mentioning me.")
                return

            await say("🧠 Thinking…")
            response = await agent.prompt(user_text)
            await say(response)

        # Handle direct messages
        @slack_app.event("message")
        async def handle_dm(event, say):
            # -- ignore anything that isn't a direct user DM -------------
            if event.get("channel_type") != "im":
                return
            if event.get("subtype") or event.get("bot_id"):
                return

            text = (event.get("text") or "").strip()
            if not text:
                return

            # Show “thinking…” indicator
            await say("🧠 Thinking…")

            # 1️⃣  Normal assistant response
            reply = await agent.prompt(text)
            await say(reply)

        # Websocket listener
        handler = AsyncSocketModeHandler(slack_app, app_token)
        await handler.start_async()       # blocks forever (Ctrl-C to quit)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\n👋 Goodbye!")
