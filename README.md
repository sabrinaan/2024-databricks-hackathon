# Databricks Hackathon 2024 - Data Science Response Bot

### Running Flask Server Locally

1. Export the Bot User OAuth Token to your local environment as `SLACK_BOT_TOKEN`.  Available [here](https://api.slack.com/apps/A06FG4L3944/install-on-team?)
2. Run `python3 slack_bot.py`
3. In a new terminal window, run `ngrok http http://127.0.0.1:5000` to tunnel your local server to the internet. [Ngrok install](https://ngrok.com/download)
4. Copy the forwarding URL to the slack bot's [event subscriptions page](https://api.slack.com/apps/A06FG4L3944/event-subscriptions?)
5. Send a message to the #databricks-hackathon-test channel

### Project Summary:
The Tatari 2024 Databricks Hackathon project is an LLM Slack Agent integrated with our company Slack channels capable of answering commonly asked questions. Our model is hosted by a Databricks Serverless Endpoint that queries the pay-per-token Databricks-hosted LLM endpoints (Llama2-70B and the embeddings endpoint). To populate our vector store we provide an Airflow DAG with a list of Confluence page_id's. The DAG then scrapes the provided Confluence page_id's (and all associated children pages); breaks the pages into chunks with associated metadata and saves everything to Amazon S3. Note: we will utilize the Databricks VectorStore for this process once our Data Platform team enables Unity Catalog.

When a question is entered into Slack the following process is followed:
- Slack sends the message to Flask, Flask sends the message to the Agent serverless endpoint
- The serverless endpoint instantiates the vector store
- The incoming question is sent to the Databricks embedding endpoint and vectorized
- The vector store is queried for the K-nearest neighbor Confluence document chunks (utilizing Maximum Marginal Relevance)
- A prompt is constructed where we include the question, returned context from the vector store, and call the Databricks Llama2-70B endpoint asking whether there is sufficient context to answer the question (and providing a JSON schema in which it must return its answer)
- If the bot cannot answer the question: we return this response to the Slack app, along with the Confluence document chunk metadata (that can then be used to improve our Confluence documentation)
- If the bot can answer the question we then construct a prompt that includes the question and the Confluence context and use the Databricks Llama2-70B endpoint to answer the question
- The answer is then routed back to Slack and posted as an in-thread response to the original question. Currently, we are using a private parallel channel to gather performance data on the Slack bot before releasing it live into the public channels.

Future planned improvements on this model include full thread memory to provide a LangChain Chatbot, backed by a fine-tuned model (utilizing the existing Slack channel data) hosted on our own Databricks Serverless GPU Endpoint.
