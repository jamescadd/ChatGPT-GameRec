FROM mambaorg/micromamba:0.15.3
USER root
RUN apt-get update && DEBIAN_FRONTEND=“noninteractive” apt-get install -y --no-install-recommends \
       nginx \
       ca-certificates \
       apache2-utils \
       certbot \
       python3-certbot-nginx \
       sudo \
       cifs-utils \
       && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update
RUN apt-get install -y libasound2
RUN apt-get install -y build-essential
RUN apt-get install -y libssl1.1
RUN apt-get install -y cron
RUN mkdir /opt/gamerec
RUN chmod -R 777 /opt/gamerec
WORKDIR /opt/gamerec
USER micromamba
COPY environment.yml environment.yml
RUN micromamba install -y -n base -f environment.yml && \
    micromamba clean --all --yes
COPY run.sh run.sh
COPY nginx.conf /etc/nginx/nginx.conf
COPY . .
USER root
RUN chmod a+x run.sh
CMD ["./run.sh"]
