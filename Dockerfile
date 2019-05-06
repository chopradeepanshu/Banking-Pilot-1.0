FROM python:3.7.1
ADD requirements.txt /
#RUN python3.7 -m pip install -r requirements.txt --proxy=http://ingssweb.ey.net:8080/
RUN python3.7 -m pip install -r requirements.txt
EXPOSE 5000

ADD ./Customer_Twitter_Data.py /
ADD ./Customer_Churn_Modelling_Light.py /
ADD ./app_1.py /
ADD ./BankCustomerChurnDataFeb19.xlsx /
ADD ./BankCustomerChurnDataFeb19_Heavy.xlsx /
ADD ./BOACustomerSentimentScore.csv /
ADD ./BOA_Customer_Tweets_And_Its_Score.csv /
ADD ./run.sh /

RUN chmod 755 /run.sh

CMD [ "/run.sh" ]