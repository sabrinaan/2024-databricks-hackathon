# Databricks Hackathon 2024 - Data Science Response Bot

### Running Flask Server Locally

1. Export the Bot User OAuth Token to your local environment as `SLACK_BOT_TOKEN`.  Available [here](https://api.slack.com/apps/A06FG4L3944/install-on-team?)
2. Run `python3 slack_bot.py`
3. In a new terminal window, run `ngrok http http://127.0.0.1:5000` to tunnel your local server to the internet. [Ngrok install](https://ngrok.com/download)
4. Copy the forwarding URL to the slack bot's [event subscriptions page](https://api.slack.com/apps/A06FG4L3944/event-subscriptions?)
5. Send a message to the #databricks-hackathon-test channel
