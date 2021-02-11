FROM python:3.8
LABEL maintainer="Moontrey"

ENV MODEL_HOME /opt/model

RUN pip install --upgrade cython
RUN pip install --upgrade pip

RUN mkdir -p ${MODEL_HOME}/scripts && \
    mkdir -p ${MODEL_HOME}/data

# ADD model/requirements.txt .
ADD requirements.txt .
RUN pip install -r requirements.txt

# COPY model/train.py ${MODEL_HOME}/scripts/train.py 
# COPY model/data_processed.csv ${MODEL_HOME}/data/data_processed.csv
COPY train.py ${MODEL_HOME}/scripts/train.py 
COPY data_processed.csv ${MODEL_HOME}/data/data_processed.csv
# CMD python opt/model/scripts/train.py 
CMD python opt/model/scripts/train.py 

# RUN \
#     wget https://dl.bintray.com/boostorg/release/1.63.0/source/boost_1_63_0.zip && \
#     unzip boost_1_63_0.zip && \
#     mv boost_1_63_0 /usr/local/bin

# RUN \
# 	git clone https://github.com/facebookresearch/Starspace.git && \
# 	cd Starspace && \
# 	make

# COPY ./uci_train_starspace_formatted.txt ./

# RUN chown :100000 ./*

# RUN \
# 	addgroup --gid 100000 test_group

# RUN \
# 	adduser --disabled-password --gecos "" --force-badname --gid 100000 test_user

# USER test_user

# CMD	\
# 	cd /home/jovyan && \ 
# 	/Starspace/starspace train -trainFile /uci_train_starspace_formatted.txt -model starspace_embeds
#	cp starspace_embeds.tsv /home/jovyan/starspace_embeds.tsv
# 	# chmod 777 starspace_embeds.tsv \ 
# 	mv starspace_embeds.tsv /home/jovyan/

# CMD touch /jasozdalsa_v2 && cp jasozdalsa_v2 /home/jovyan/

