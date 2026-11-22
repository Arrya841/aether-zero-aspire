FROM python:3.13-slim
WORKDIR /bot
COPY mechfight.py bot.py policy.py ground_map.py combat_policy.py ./
ENTRYPOINT ["python3", "-B", "-u", "/bot/bot.py"]
