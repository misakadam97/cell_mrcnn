FROM continuumio/miniconda3
ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
RUN mkdir /home/mrcnn 
WORKDIR /home/mrcnn
COPY . .
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH
SHELL ["conda", "run", "-n", "cell_mrcnn", "/bin/bash", "-c"]
RUN python /home/mrcnn/setup.py install
RUN pip install streamlit
RUN pip install scikit-image
CMD streamlit run app/cell_app.py --server.port 6123

