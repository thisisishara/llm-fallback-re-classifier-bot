# llm_fallback_re_classifier-bot
A rasa bot that uses a custom NLU pipeline component that leverages LLMs to re-classify nlu_fallback intents predicted by default, using LLMs.

## Bot Domain
For demo purposes, the bot is initialized as a assistant of a Pizza Shop. You can ask questions about the menu, pricing, ordering, and also say greet, thanks, and goodbye.

## Deployment
### Local
If the bot is deployed locally, make sure to set the `OPENAI_COMPLETIONS_API_KEY` environment variable with your open AI API access token.
- Set the environment variable
- Train a new model using `rasa train`
- Run the rasa bot by running `rasa run --enable-api --cors "*"`

### Docker Compose  
Docker compose configs are provided to easily run the bot.
- Make sure `.env` contains the open AI API access token.
- Run `docker compose up -d`, a new rasa model will be trained when the image is building.
- If you want to rebuild everything, run `docker compose up --build -d`.

## Inference
To see how the `llm_fallback_re_classifier` is re-classifying fallback intents, simply inspect the logs and chat with the bot with complex queries.
- If deployed locally, use a tool like postman and send POST requests to the `http://localhost/core/model/parse` endpoint. 
Request body: 
```json
{
  "text": "your_query_here"
}
```

- If deployed with docker, inspect the logs and open the chat widget by visiting `http://localhost` and chat with the bot.

## What is  ⚡LLMFallbackReClassifier custom component?
It is an extended version of the default `FallbackClassifier` available in Rasa that takes further steps to attempt to re-classify user queries where `nlu_fallback` intent is predicted due to law confidence of all top ranking intents.  

It asks the LLM to attempt to classify the user query into one of the top ranking intents available in `intent_ranking` list or else classify it under `nlu_fallback` if it cannot be done. The prompt can be further fine-tuned to suit specific use cases.


### Example
🤔 Without LLMFallbackReClassifier, the output of the following query is a fallback for the given bot.
query:
```json
{
    "text": "I want to look at the prices, but wait, show me the menu instead"
}
```
output:
```json
{
    "text": "I want to look at the prices, but wait, show me the menu instead",
    "intent": {
        "name": "nlu_fallback",
        "confidence": 0.9
    },
    "entities": [],
    "text_tokens": []
    "intent_ranking": [
        {
            "name": "nlu_fallback",
            "confidence": 0.9
        },
        {
            "name": "request_prices",
            "confidence": 0.7004178166389465
        },
        {
            "name": "request_menu",
            "confidence": 0.2550094723701477
        },
        {
            "name": "place_order",
            "confidence": 0.03543368726968765
        },
        {
            "name": "deny",
            "confidence": 0.006627656519412994
        },
        {
            "name": "say_greet",
            "confidence": 0.0012468327768146992
        },
        {
            "name": "affirm",
            "confidence": 0.0010363396722823381
        },
        {
            "name": "say_goodbye",
            "confidence": 0.00019269764015916735
        },
        {
            "name": "say_thanks",
            "confidence": 3.550011388142593e-5
        }
    ]
}
```


✅ With LLMFallbackReClassifier, the output of the above query is corrected by the LLM as follows.
query:
```json
{
    "text": "I want to look at the prices, but wait, show me the menu instead"
}
```
output:
```json
{
    "text": "I want to look at the prices, but wait, show me the menu instead",
    "intent": {
        "name": "request_menu",
        "confidence": 0.9
    },
    "entities": [],
    "text_tokens": []
    "intent_ranking": [
        {
            "name": "request_menu",
            "confidence": 0.9
        },
        {
            "name": "request_prices",
            "confidence": 0.7004178166389465
        },
        {
            "name": "place_order",
            "confidence": 0.03543368726968765
        },
        {
            "name": "deny",
            "confidence": 0.006627656519412994
        },
        {
            "name": "say_greet",
            "confidence": 0.0012468327768146992
        },
        {
            "name": "affirm",
            "confidence": 0.0010363396722823381
        },
        {
            "name": "say_goodbye",
            "confidence": 0.00019269764015916735
        },
        {
            "name": "say_thanks",
            "confidence": 3.550011388142593e-5
        }
    ]
}
```
