#base image
FROM python:3.9
#copy everything from current dir (.) to work dir
RUN pip install --upgrade pip
RUN pip install cryptography
RUN pip install --no-dependencies transformers
# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
COPY . /app
WORKDIR /app
ENV CRYPTOGRAPHY_DONT_BUILD_RUST=1
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]

