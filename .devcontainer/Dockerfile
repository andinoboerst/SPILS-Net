FROM --platform=linux/amd64 dolfinx/dolfinx:v0.9.0-arm64

RUN apt-get update && \
      apt-get -y install sudo

RUN apt-get install --fix-missing -y xvfb

RUN pip install --upgrade pip

RUN pip install matplotlib pandas pyvista[jupyter] trame trame-vtk jupyter imageio imageio[ffmpeg] scikit-learn torch lion_pytorch torchvision ray[tune] pygmsh

# Set the working directory to /workspace
WORKDIR /workspace
