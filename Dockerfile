# Start from a base image that includes Python and Jupyter
FROM jupyter/base-notebook:latest

# Switch to the root user to install system packages
USER root

# Tell the container to use the bash shell for commands
SHELL ["/bin/bash", "-c"]

# Update package lists and install the command for adding new repositories
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    # Add the official FEniCS package repository (PPA)
    add-apt-repository ppa:fenics-packages/fenics -y && \
    # Update the package lists again to include the new FEniCS packages
    apt-get update && \
    # Now, install fenics, which the system can now find
    apt-get install -y --no-install-recommends fenics && \
    # Clean up the package manager cache to keep the image smaller
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch back to the default, non-root user that Binder expects
USER ${NB_UID}
