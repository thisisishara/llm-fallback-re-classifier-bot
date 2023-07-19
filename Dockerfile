FROM rasa/rasa:3.6.2-full@sha256:32c6d56b6a5fe8a2035b83b5c04ba88715843132f86ef5f6c9e41decc149f8cc

# Change back to root user to install dependencies
USER root

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY data /app/data
COPY custom /app/custom
COPY *.yml .

RUN rasa train

# Switch back to non-root to run code
USER 1001

EXPOSE 5005
CMD ["run", "--enable-api", "--cors", "*"]
