# docker build -t cpc4ad:vX .
# docker run -it cpc4ad:vX
FROM pytorch-with-lib

WORKDIR /

COPY . .

RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 5000

CMD python anomaly_detection.py
