FROM centos:centos7

RUN yum update -y
RUN yum install -y scl-utils centos-release-scl epel-release gcc wget
RUN yum install -y java-1.8.0-openjdk python27 python-pip
RUN scl enable python27 bash
RUN pip install --upgrade pip

RUN wget http://people.tuke.sk/martin.sarnovsky/tsvd/files/spark-2.2.0-bin-hadoop2.6.tgz && tar -xzf spark-2.2.0-bin-hadoop2.6.tgz && rm spark-2.2.0-bin-hadoop2.6.tgz
COPY template /spark-2.2.0-bin-hadoop2.6/conf/log4j.properties

ENV SPARK_HOME=/spark-2.2.0-bin-hadoop2.6
ENV PATH=$PATH:$SPARK_HOME/bin
RUN ln -s $SPARK_HOME /opt/spark

## DEV
COPY /python/requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR zadanie/python
ENTRYPOINT ["bash"]

## PRODUCTION
# COPY /data /data
# COPY /python /python

# WORKDIR python
# RUN pip install -r requirements.txt
# ENTRYPOINT ["spark-submit", "--master local[2]", "data_load.py"]