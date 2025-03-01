FROM python:3.12

EXPOSE 8080

WORKDIR /app

COPY . ./

RUN pip3 install --no-cache-dir -r requirements.txt \
    && mkdir -p ~/.streamlit && echo "\
    [server]\n\
    headless = true\n\
    enableCORS = false\n\
    port = 8080\n\
    " > ~/.streamlit/config.toml

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]