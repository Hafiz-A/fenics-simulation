# Start from a base image that includes Python and Jupyter
FROM jupyter/base-notebook:latest

# Switch to the root user to install system packages
USER root

# Update package lists and install FEniCS directly from the default repository
RUN apt-get update && \
    apt-get install -y --no-install-recommends fenics && \
    # Clean up the package manager cache to keep the image smaller
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch back to the default, non-root user that Binder expects
USER ${NB_UID}
